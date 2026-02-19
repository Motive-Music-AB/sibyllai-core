# Kazen Design System — Motive (2026)

A comprehensive design specification for the Motive music discovery application. This document captures every visual decision: color tokens, typography, glass surfaces, spacing, components, and layout architecture.

---

## 1. Design Philosophy

- **Aesthetic**: High-contrast glassmorphism over photographic backgrounds. Forced dark mode only.
- **Atmosphere**: Premium, cinematic, layered. Glass surfaces float over a fixed landscape photograph creating depth and parallax.
- **Tone**: Professional creative tooling — not playful, not corporate. Feels like a high-end audio workstation.
- **Differentiator**: The warm amber accent against cool, frosted glass creates a signature look. No generic purple-gradient AI aesthetic.

---

## 2. Color System

All colors are defined as HSL values in CSS custom properties (`:root`). Tailwind references them via `hsl(var(--token))`.

### 2.1 Core Palette

| Token | HSL Value | Usage |
|-------|-----------|-------|
| `--background` | `0 0% 10%` | Page background / scrim base |
| `--foreground` | `0 0% 93%` | Primary text — near-white |
| `--primary` | `35 80% 55%` | Warm amber — buttons, accents, logo, waveform active |
| `--primary-foreground` | `0 0% 7%` | Text on primary surfaces — near-black |
| `--accent` | `35 80% 55%` | Same as primary (unified accent) |
| `--accent-foreground` | `0 0% 7%` | Text on accent surfaces |
| `--destructive` | `0 62% 55%` | Error / destructive actions |
| `--destructive-foreground` | `0 0% 100%` | Text on destructive surfaces |

### 2.2 Secondary & Muted

| Token | HSL Value | Usage |
|-------|-----------|-------|
| `--secondary` | `0 0% 16%` | Secondary surface fills |
| `--secondary-foreground` | `0 0% 85%` | Secondary text — near-white, slightly softer |
| `--muted` | `0 0% 18%` | Muted surface fills |
| `--muted-foreground` | `0 0% 82%` | Muted text — near-white, NOT grey. Readable always. |

> **Rule**: No grey text inside glass panels. All body/label text is white or near-white (`82–93%` lightness). Only elements _outside_ glass (e.g. top bar tagline, back button) may use dark/black text to contrast against the bright background image.

### 2.3 Surfaces (Glass Layers)

| Token | HSL Value | Usage |
|-------|-----------|-------|
| `--surface-0` | `0 0% 10%` | Deepest layer (matches background) |
| `--surface-1` | `0 0% 14%` | First elevation |
| `--surface-2` | `0 0% 18%` | Second elevation |
| `--surface-3` | `0 0% 22%` | Third elevation (card fills) |
| `--surface-hover` | `0 0% 100% / 0.06` | Hover state overlay |

### 2.4 Borders & Input

| Token | HSL Value | Usage |
|-------|-----------|-------|
| `--border` | `0 0% 100% / 0.08` | Default border — subtle white at 8% |
| `--input` | `0 0% 100% / 0.08` | Input field borders |
| `--ring` | `35 80% 55%` | Focus ring — matches primary amber |

### 2.5 Waveform-Specific

| Token | HSL Value | Usage |
|-------|-----------|-------|
| `--waveform` | `35 80% 55%` | Active waveform bars — amber |
| `--waveform-inactive` | `0 0% 30%` | Inactive/unselected waveform bars |
| `--waveform-selected` | `35 90% 62%` | Selected segment — brighter amber |
| `--waveform-bg` | `0 0% 8%` | Waveform container background |

### 2.6 Glow & Glass Tokens

| Token | HSL Value | Usage |
|-------|-----------|-------|
| `--glow` | `35 80% 55%` | Glow effects — amber |
| `--glass-bg` | `0 0% 16%` | Glass background base |
| `--glass-border` | `0 0% 100%` | Glass border base (applied with opacity) |
| `--glass-blur` | `30px` | Default blur radius for glass |

### 2.7 Card & Popover

| Token | HSL Value | Usage |
|-------|-----------|-------|
| `--card` | `0 0% 22%` | Card background |
| `--card-foreground` | `0 0% 93%` | Card text |
| `--popover` | `0 0% 16%` | Popover/dropdown background |
| `--popover-foreground` | `0 0% 93%` | Popover text |

---

## 3. Typography

### 3.1 Font Stack

| Role | Font Stack | Weight |
|------|-----------|--------|
| **Display / Headings** | `'Geist', 'Inter', system-ui, sans-serif` | 600 (semibold) |
| **Body / UI** | `'Geist', 'Inter', system-ui, -apple-system, sans-serif` | 300–500 |
| **Monospace / Data** | `'Geist Mono', 'JetBrains Mono', ui-monospace, monospace` | 400 |

### 3.2 Scale & Usage

| Element | Size | Weight | Color | Notes |
|---------|------|--------|-------|-------|
| Logo ("Motive") | `text-6xl` (3.75rem) | 600 | `text-primary` | `drop-shadow-lg`, `.font-display` |
| Tagline (top bar) | `13px` | 500 | `text-black` | Uppercase, `tracking-[0.25em]` |
| Back button | `13px` | 500 | `text-black` | Black for contrast on bright bg |
| Section titles (Cues, Matches, Details) | `text-sm` (14px) | 600 | `text-foreground` | `.font-display` |
| Track title | `text-lg` (18px) | 600 | `text-foreground` | `.font-display` |
| Body text / values | `11–13px` | 400–500 | `text-foreground` | Near-white |
| Labels (BPM, Key, etc.) | `9px` | 400 | `text-muted-foreground` | Uppercase, `tracking-wider` |
| Metadata / timestamps | `9–11px` | 400 | `text-muted-foreground` | `.font-mono` |
| Timecode | `12px` | 400 | `text-foreground` | `.font-mono`, `tracking-wide` |
| Time ruler | `9px` | 400 | `text-muted-foreground` | `.font-mono` |

### 3.3 Text Color Rules

1. **Inside glass panels**: All text is white (`text-foreground`, 93% lightness) or near-white (`text-muted-foreground`, 82% lightness). Never grey.
2. **Outside glass panels** (top bar over background image): Use `text-black` for non-accent text to contrast against the bright photographic background.
3. **Accent text**: Use `text-primary` (amber) for the logo, active states, matched badges, and interactive highlights.

---

## 4. Glass & Surface System

### 4.1 Hierarchy

The UI uses a layered glass system. Each level adds opacity and blur to create depth:

```
Background Image (fixed, full-bleed)
  └─ .glass-panel (main container — blurs background)
       └─ .glass-card (inner content pieces — blurs panel)
            └─ .panel-inset (recessed areas within cards)
```

### 4.2 Glass Panel (`.glass-panel`)

The outermost glass container. Wraps the entire main content area.

```css
background: hsla(0, 0%, 20%, 0.55);     /* Bright grey at 55% opacity */
backdrop-filter: blur(30px);              /* Frosts the background image */
border: 1px solid hsl(0 0% 100% / 0.1);  /* Subtle white edge */
border-radius: 2rem;                      /* 32px — large radius */
```

### 4.3 Glass Card (`.glass-card`)

Inner content pieces within a glass panel (cue cards, info sections, toggle groups).

```css
background: hsl(0 0% 100% / 0.07);       /* White at 7% opacity */
backdrop-filter: blur(20px);              /* Secondary blur layer */
border: 1px solid hsl(0 0% 100% / 0.08); /* Faint white edge */
border-radius: 1rem;                      /* 16px */
transition: background 0.2s ease;
```

### 4.4 Panel Elevated (`.panel-elevated`)

Used for the waveform timeline — higher elevation with stronger blur and shadow.

```css
background: hsla(0, 0%, 18%, 0.55);
backdrop-filter: blur(40px);
border: 1px solid hsl(0 0% 100% / 0.1);
border-radius: 2rem;
box-shadow:
  0 12px 40px hsl(0 0% 0% / 0.3),
  inset 0 1px 0 hsl(0 0% 100% / 0.06);
```

### 4.5 Panel Inset (`.panel-inset`)

Recessed areas (waveform display region).

```css
background: hsl(0 0% 100% / 0.03);
border: 1px solid hsl(0 0% 100% / 0.04);
border-radius: 1rem;
```

---

## 5. Border Radius System

| Token | Value | Usage |
|-------|-------|-------|
| `--radius` | `1rem` (16px) | Default border radius |
| `--radius-panel` | `2rem` (32px) | Outer glass panels |
| `--radius-card` | `1rem` (16px) | Inner cards |
| Tailwind `lg` | `var(--radius)` | 16px |
| Tailwind `md` | `calc(var(--radius) - 2px)` | 14px |
| Tailwind `sm` | `calc(var(--radius) - 4px)` | 12px |

**Rule**: Always use high-radius surfaces. Panels are 32px, cards are 16px, pills/badges are `rounded-full`.

---

## 6. Spacing & Layout

### 6.1 Page Structure

- **Max content width**: `1480px`, centered with `mx-auto`
- **Page padding**: `px-6 pb-6 pt-2`
- **Glass panel internal padding**: `p-6`
- **Section gap** (within glass panel): `space-y-5` (20px)
- **Bottom grid**: 3-column equal grid, `gap-4` (16px)

### 6.2 Component Spacing

| Context | Value |
|---------|-------|
| Top bar padding | `px-8 py-5` |
| Track header padding | `px-6 py-5` |
| Waveform panel padding | `p-5` |
| Info panel padding | `p-5` |
| Card internal gap | `space-y-4` |
| Icon + text gap | `gap-2.5` to `gap-4` |
| Label + value gap | `gap-2` |

---

## 7. Background System

### 7.1 Fixed Background Image

The entire app sits over a fixed, full-bleed photographic background:

```tsx
<div className="fixed inset-0 pointer-events-none">
  <img
    src="/images/kazen-bg-3.png"
    alt=""
    className="absolute inset-0 w-full h-full object-cover"
  />
</div>
```

- **File**: `/images/kazen-bg-3.png` — warm landscape photograph (mountains/sunset tones)
- **Positioning**: `fixed inset-0`, `object-cover`
- **No overlay/scrim**: The glass panels provide their own contrast via blur and opacity. No dark overlay is applied over the image.

---

## 8. Component Patterns

### 8.1 Primary Button (`.btn-primary`)

```css
background: hsl(var(--primary));              /* Amber fill */
color: hsl(var(--primary-foreground));        /* Near-black text */
box-shadow:
  0 2px 8px hsl(var(--primary) / 0.3),        /* Amber glow */
  inset 0 1px 0 hsl(0 0% 100% / 0.15);       /* Top edge highlight */
```

Hover: slightly darker amber (`hsl(35 80% 50%)`), stronger glow, `scale-[1.02]`.

Used as pills: `rounded-full`, `px-5 py-1.5` or `px-3.5 py-1.5`, `text-[12px] font-medium`.

### 8.2 Badges

- **Status badge** (e.g., "MATCHED"): `bg-primary/10 text-primary border border-primary/15 rounded-full`, `text-[9px] uppercase tracking-[0.08em] font-semibold`
- **Segment labels** (e.g., "Select", "Selected"): `rounded-full surface-3 text-muted-foreground border border-border/30`, `text-[9px] font-medium`
- **Selected segment badge**: `bg-primary/15 text-primary border-primary/25` with a checkmark SVG icon

### 8.3 Icon Containers

Small icon squares used for track/cue identifiers:

```
w-8 h-8 (or w-10 h-10) rounded-lg (or rounded-xl)
bg-primary/10 border border-primary/20
```

Icon size: `w-3.5 h-3.5` to `w-4 h-4`, color `text-primary`.

### 8.4 Toggle Group (Match / Library)

Pill-shaped container using `.glass-card rounded-full p-1`. Active tab uses `.btn-primary`, inactive tab uses `text-muted-foreground hover:text-foreground`.

### 8.5 Transport Controls

- **Timecode display**: `.font-mono text-[12px] surface-2 px-3 py-2 rounded-lg border border-border/40`
- **Playhead**: `w-px bg-primary/50` vertical line with a `w-2 h-2 rounded-full bg-primary glow-sm` dot at top
- **Activity dots**: 4× `w-1.5 h-1.5 rounded-full bg-primary animate-pulse-dot` with staggered delays

---

## 9. Glow Effects

| Class | Effect |
|-------|--------|
| `.glow-primary` | `0 0 20px amber/0.2, 0 0 60px amber/0.08` — dramatic outer glow |
| `.glow-sm` | `0 0 12px amber/0.15` — subtle glow (playhead dot) |
| `.btn-primary` hover | `0 4px 16px amber/0.4` — button press glow |

---

## 10. Animations

| Name | Duration | Easing | Usage |
|------|----------|--------|-------|
| `pulse-dot` | 2s | ease-in-out, infinite | Transport activity dots. Scales 1→1.2, opacity 0.3→1 |
| `shimmer` | 3s | ease-in-out, infinite | Loading shimmer effect |
| `accordion-down/up` | 0.2s | ease-out | Collapsible panels |

### Transitions

- **Glass card hover**: `background 0.2s ease`
- **Text color hover**: `transition-colors duration-200`
- **Button hover**: `transition-all duration-200`
- **Waveform bars**: `transition-all duration-300`

---

## 11. Waveform Visualization

### Bar Rendering

Each bar is a `2px`-wide rounded rectangle with a mirrored reflection:

```
Main bar:  w-[2px] rounded-full, height varies (sine + random)
Reflection: w-[2px] rounded-full, 25% of main height, opacity-30
```

### Color States

| State | Color Class | Token |
|-------|-------------|-------|
| Active | `bg-waveform` | `35 80% 55%` (amber) |
| Selected | `bg-waveform-selected` | `35 90% 62%` (bright amber) |
| Inactive | `bg-waveform-inactive` | `0 0% 30%` (dark grey) |

### Selected Segment

Highlighted with `bg-primary/[0.06]` fill and `border border-primary/20` overlay.

---

## 12. Sidebar Tokens (Reserved)

| Token | HSL Value |
|-------|-----------|
| `--sidebar-background` | `0 0% 12%` |
| `--sidebar-foreground` | `0 0% 55%` |
| `--sidebar-primary` | `35 80% 55%` |
| `--sidebar-primary-foreground` | `0 0% 7%` |
| `--sidebar-accent` | `0 0% 16%` |
| `--sidebar-accent-foreground` | `0 0% 60%` |
| `--sidebar-border` | `0 0% 100% / 0.08` |
| `--sidebar-ring` | `35 80% 55%` |

---

## 13. Key Design Rules

1. **No grey text inside panels.** All text within glass surfaces is white or near-white.
2. **Black text only outside glass** — for elements sitting directly on the background image (top bar tagline, back button).
3. **Amber is the only accent color.** It appears in: logo, buttons, waveform, badges, glows, playhead, focus rings.
4. **All surfaces use backdrop-filter blur.** Every glass layer frosts what's beneath it.
5. **No dark overlay on the background image.** Glass panels provide their own contrast.
6. **High radius everywhere.** 32px panels, 16px cards, full-round pills.
7. **Semantic tokens only.** Components never use raw color values — everything goes through CSS variables and Tailwind's token system.
8. **Forced dark mode.** No light mode variant exists. The `:root` scope defines the single theme.
