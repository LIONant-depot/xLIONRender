#ifndef XLIONRENDER_TEXT_LAYOUT_H
#define XLIONRENDER_TEXT_LAYOUT_H
#pragma once

// The layout of a Text (xlionrender_text.h): where every glyph goes. Pure computation over the compiled data of a Font resource - no GPU, no ECS - so it is the same for every encoding the font
// can have (MTSDF, SDF, BITMAP) and testable on its own. The result is in em units: the renderer scales it to the world or to pixels, which is why a change of the entity's transform, of the
// camera or of the size mode never makes a layout again.
//
//   - Codepoints: the text is UTF-16 on Windows (std::wstring): a surrogate pair is one codepoint (an emoji is not two characters). The font finds a glyph by codepoint (a perfect hash); a
//     codepoint it does not have is drawn as U+FFFD or '?' if the font has either, and as nothing (counted in m_nMissing) if it has neither. There is no shaping: no right-to-left text, no
//     combining marks, no ligatures - the font resource has none of that.
//   - Metrics: the advance and the bearings of the glyphs and the ascender, descender and line height of the font - never the size of the rectangle of a glyph in the atlas.
//   - Kerning: when the font has it (MTSDF and SDF); a BITMAP font has none baked, so its spacing is the plain advances.
//   - BITMAP: the glyphs of the baked size closest to the size the text will be seen at (m_PixelSize), or the biggest when that is not known; its metrics are in em like every other encoding.
#include "xlionrender_text.h"
#include "plugins/xfont.plugin/source/xfont_rsc_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <vector>

namespace xlionrender::text_layout
{
    // One glyph, ready for the GPU: the corners in em units (y up, the origin of the entity at the anchor of the text) and the rectangle of the atlas (normalized). The quad is a half texel bigger
    // on every side than the glyph, and so is its rectangle, so the filter of the sampler finds the edge of the glyph instead of the neighbour (the same as the font editor's preview).
    struct glyph_quad
    {
        xmath::fvec2    m_Min, m_Max;
        xmath::fvec2    m_UVMin, m_UVMax;
    };

    struct settings
    {
        float           m_WrapEm        = 0.0f;                     // 0: only the line breaks of the text break lines
        float           m_LineSpacing   = 1.0f;                     // times the line height of the font
        text_halign     m_HAlign        = text_halign::CENTER;
        text_vanchor    m_VAnchor       = text_vanchor::MIDDLE;
        float           m_PixelSize     = 0.0f;                     // BITMAP: the size in pixels one em will be seen at; 0 = the biggest baked size
        int             m_AtlasWidth    = 0;                        // pixels of the atlas texture (the rectangles of the glyphs are in pixels)
        int             m_AtlasHeight   = 0;
    };

    struct result
    {
        std::vector<glyph_quad> m_Quads;
        xmath::fvec2            m_Min   = {};                       // the box of the text by its metrics (not by the ink): the width of the lines, the height of the lines. em units, origin at the anchor
        xmath::fvec2            m_Max   = {};
        int                     m_nLines    = 0;
        int                     m_nMissing  = 0;                    // codepoints the font has no glyph for (and no replacement)
    };

    // The codepoints of the text. UTF-16 on Windows: a high surrogate followed by a low surrogate is one codepoint, anything else is one codepoint by itself.
    inline void Decode(std::wstring_view Text, std::vector<std::uint32_t>& Out) noexcept
    {
        Out.clear();
        Out.reserve(Text.size());
        for (std::size_t i = 0; i < Text.size(); ++i)
        {
            const std::uint32_t Unit = static_cast<std::uint32_t>(Text[i]);
            if constexpr (sizeof(wchar_t) == 2)
            {
                if (Unit >= 0xD800 && Unit <= 0xDBFF && i + 1 < Text.size())
                {
                    const std::uint32_t Low = static_cast<std::uint32_t>(Text[i + 1]);
                    if (Low >= 0xDC00 && Low <= 0xDFFF) { Out.push_back(0x10000 + ((Unit - 0xD800) << 10) + (Low - 0xDC00)); ++i; continue; }
                }
            }
            Out.push_back(Unit);
        }
    }

    namespace details
    {
        struct placed
        {
            const xfont_rsc::glyph* m_pGlyph  = nullptr;
            std::uint32_t           m_Codepoint = 0;
            float                   m_X       = 0.0f;               // the pen position in the line (em)
            float                   m_Advance = 0.0f;
        };

        struct line
        {
            std::vector<placed>     m_Glyphs;
            float                   m_Width = 0.0f;                 // up to the end of the last glyph that is not a space
        };

        // The baked size a BITMAP font is read at: the closest to what was asked, the biggest when nothing was.
        inline const xfont_rsc::size_group_header* PickSizeGroup(const xfont_rsc::font& Font, float PixelSize) noexcept
        {
            if (Font.m_nSizeGroups == 0) return nullptr;
            if (PixelSize > 0.0f) return Font.FindClosestSizeGroup(PixelSize);
            const xfont_rsc::size_group_header* pBiggest = &Font.SizeGroups()[0];
            for (std::uint32_t i = 1; i < Font.m_nSizeGroups; ++i)
                if (Font.SizeGroups()[i].m_PixelSize > pBiggest->m_PixelSize) pBiggest = &Font.SizeGroups()[i];
            return pBiggest;
        }

        inline void Finish(line& Line) noexcept
        {
            Line.m_Width = 0.0f;
            for (auto It = Line.m_Glyphs.rbegin(); It != Line.m_Glyphs.rend(); ++It)
                if (It->m_Codepoint != ' ') { Line.m_Width = It->m_X + It->m_Advance; break; }
        }
    }

    inline void Layout(const xfont_rsc::font& Font, std::wstring_view Text, const settings& Settings, result& Out) noexcept
    {
        using namespace details;
        Out = {};

        // The metrics of the font. A font that says nothing about them (or says them backwards) still gets a sane line.
        float Ascender   = Font.m_Ascender;
        float Descender  = -std::abs(Font.m_Descender);
        float LineHeight = Font.m_LineHeight;
        if (Ascender <= 0.0f)   Ascender   = 0.8f;
        if (LineHeight <= 0.0f) LineHeight = Ascender - Descender;
        LineHeight *= Settings.m_LineSpacing;

        const xfont_rsc::size_group_header* pGroup = PickSizeGroup(Font, Settings.m_PixelSize);
        const char* pRegion = pGroup ? Font.SizeGroupRegionStart() : nullptr;
        const auto Find = [&](std::uint32_t Codepoint) noexcept -> const xfont_rsc::glyph*
        {
            return pGroup ? xfont_rsc::font::FindGlyphInSizeGroup(pRegion, *pGroup, Codepoint) : Font.FindGlyph(Codepoint);
        };
        const auto Glyph = [&](std::uint32_t Codepoint) noexcept -> const xfont_rsc::glyph*
        {
            if (const auto* pGlyph = Find(Codepoint)) return pGlyph;
            if (const auto* pGlyph = Find(0xFFFD))    return pGlyph;
            return Find('?');
        };

        std::vector<std::uint32_t> Codepoints;
        Decode(Text, Codepoints);

        //
        // The lines: the glyphs of each with their pen positions, broken at the line breaks of the text and, when there is a wrap width, at spaces (before the glyph that would not fit when there
        // is no space in the line)
        //
        std::vector<line> Lines(1);
        float         Pen  = 0.0f;
        std::uint32_t Prev = 0;
        for (const std::uint32_t Cp : Codepoints)
        {
            if (Cp == '\r') continue;
            if (Cp == '\n') { Finish(Lines.back()); Lines.emplace_back(); Pen = 0.0f; Prev = 0; continue; }

            const xfont_rsc::glyph* pGlyph = Glyph(Cp);
            if (!pGlyph) { ++Out.m_nMissing; Prev = 0; continue; }

            const float Kern    = (Prev != 0 && !pGroup) ? xfont_rsc::FromFixed(Font.FindKernAdjust(Prev, Cp)) : 0.0f;
            const float Advance = xfont_rsc::FromFixed(pGlyph->m_Advance);
            float       X       = Pen + Kern;

            auto& Current = Lines.back();
            if (Settings.m_WrapEm > 0.0f && !Current.m_Glyphs.empty() && Cp != ' ' && X + Advance > Settings.m_WrapEm)
            {
                // the last space of this line: what comes after it goes to the next one
                std::size_t Space = Current.m_Glyphs.size();
                for (std::size_t i = Current.m_Glyphs.size(); i-- > 0;)
                    if (Current.m_Glyphs[i].m_Codepoint == ' ') { Space = i; break; }

                line Next;
                if (Space != Current.m_Glyphs.size())
                {
                    const float Shift = Space + 1 < Current.m_Glyphs.size() ? Current.m_Glyphs[Space + 1].m_X : Pen;
                    for (std::size_t i = Space + 1; i < Current.m_Glyphs.size(); ++i)
                    {
                        placed P = Current.m_Glyphs[i];
                        P.m_X -= Shift;
                        Next.m_Glyphs.push_back(P);
                    }
                    Current.m_Glyphs.resize(Space + 1);
                }
                Finish(Current);
                Lines.push_back(std::move(Next));
                const auto& Moved = Lines.back().m_Glyphs;
                Pen = Moved.empty() ? 0.0f : Moved.back().m_X + Moved.back().m_Advance;
                X   = Pen;                                                  // nothing is kerned across a break
            }

            Lines.back().m_Glyphs.push_back({ pGlyph, Cp, X, Advance });
            Pen  = X + Advance;
            Prev = Cp;
        }
        Finish(Lines.back());
        Out.m_nLines = static_cast<int>(Lines.size());

        //
        // The block: as wide as the wrap width (or as the widest line), as tall as its lines; the alignment sets where each line sits in the width and where the origin of the entity is in it (the
        // left edge, the middle, the right edge); the anchor sets the origin vertically
        //
        float Width = Settings.m_WrapEm;
        if (Width <= 0.0f) for (const auto& Line : Lines) Width = std::max(Width, Line.m_Width);

        const float PivotX = Settings.m_HAlign == text_halign::LEFT ? 0.0f : Settings.m_HAlign == text_halign::CENTER ? Width * 0.5f : Width;
        const float Height = (Ascender - Descender) + static_cast<float>(Lines.size() - 1) * LineHeight;
        const float ShiftY = Settings.m_VAnchor == text_vanchor::TOP ? 0.0f : Settings.m_VAnchor == text_vanchor::MIDDLE ? Height * 0.5f : Settings.m_VAnchor == text_vanchor::BOTTOM ? Height : Ascender;

        Out.m_Min = { -PivotX,        ShiftY - Height };
        Out.m_Max = { Width - PivotX, ShiftY };

        //
        // The quads
        //
        const float AtlasW = static_cast<float>(Settings.m_AtlasWidth), AtlasH = static_cast<float>(Settings.m_AtlasHeight);
        if (AtlasW <= 0.0f || AtlasH <= 0.0f) return;

        for (std::size_t iLine = 0; iLine < Lines.size(); ++iLine)
        {
            const auto& Line = Lines[iLine];
            const float Offset   = Settings.m_HAlign == text_halign::LEFT ? 0.0f : Settings.m_HAlign == text_halign::CENTER ? (Width - Line.m_Width) * 0.5f : Width - Line.m_Width;
            const float Baseline = -Ascender - static_cast<float>(iLine) * LineHeight + ShiftY;

            for (const auto& P : Line.m_Glyphs)
            {
                const auto& G = *P.m_pGlyph;
                if (G.m_AtlasW == 0 || G.m_AtlasH == 0) continue;           // a space: it advances, it has no ink

                const float Left = xfont_rsc::FromFixed(G.m_PlaneLeft),  Right = xfont_rsc::FromFixed(G.m_PlaneRight);
                const float Bottom = xfont_rsc::FromFixed(G.m_PlaneBottom), Top = xfont_rsc::FromFixed(G.m_PlaneTop);
                const float HalfTexelX = 0.5f * (Right - Left) / G.m_AtlasW;
                const float HalfTexelY = 0.5f * (Top - Bottom) / G.m_AtlasH;
                const float X = Offset + P.m_X - PivotX;

                glyph_quad Q;
                Q.m_Min   = { X + Left  - HalfTexelX, Baseline + Bottom - HalfTexelY };
                Q.m_Max   = { X + Right + HalfTexelX, Baseline + Top    + HalfTexelY };
                Q.m_UVMin = { (G.m_AtlasX - 0.5f) / AtlasW, (G.m_AtlasY + G.m_AtlasH + 0.5f) / AtlasH };
                Q.m_UVMax = { (G.m_AtlasX + G.m_AtlasW + 0.5f) / AtlasW, (G.m_AtlasY - 0.5f) / AtlasH };
                Out.m_Quads.push_back(Q);
            }
        }
    }
}

#endif // XLIONRENDER_TEXT_LAYOUT_H
