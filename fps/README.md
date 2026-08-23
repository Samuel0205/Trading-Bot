# Breach Protocol — a raycasting first-person shooter

A single-file, no-dependency browser FPS built on the same engine lineage as the
original Doom: **raycasting**. Real 3D perspective, textured walls, sprite
enemies, multiple weapons, and a per-column depth buffer so enemies are correctly
hidden behind walls. Everything — wall textures, enemies, weapon models, sound —
is generated procedurally in code. No images, no audio files, no build step.

## Play it

Open `fps/index.html` in any modern browser.

```
open fps/index.html        # macOS
xdg-open fps/index.html    # Linux
start fps\index.html       # Windows
```

## Controls

**Desktop**
| Action        | Input                                   |
|---------------|-----------------------------------------|
| Move          | `W` `A` `S` `D` / arrow keys            |
| Look          | Mouse (click the screen to lock it; `Esc` releases) |
| Fire          | Left mouse (hold for automatic weapons) |
| Switch weapon | `1` pistol · `2` rifle · `3` shotgun · `4` plasma |

**Touch:** left virtual stick to move, drag the right side of the screen to look,
**FIRE** button to shoot, **SWAP** to cycle weapons.

## The game

- You spawn in a breached reactor facility. Hostiles pour in each **wave**.
- **Hunt and kill** every hostile to clear the wave; each wave is bigger and faster.
- Enemies drop **health**, **armor**, **ammo**, and locked **weapons** (shotgun, plasma).
- **Armor** absorbs half of incoming damage until it runs out. Health doesn't regen
  except a small patch between waves — play carefully.
- Minimap (top-right) shows walls, hostiles (red), pickups (green) and your heading.

## Weapons

| # | Weapon  | Feel                                             |
|---|---------|--------------------------------------------------|
| 1 | Pistol  | Infinite ammo, reliable, semi-auto               |
| 2 | Rifle   | Fast full-auto, your workhorse                   |
| 3 | Shotgun | 8 pellets, devastating up close, slow            |
| 4 | Plasma  | High-damage full-auto energy                     |

## What this is (and isn't)

This is a genuine, playable first-person shooter using the **raycasting** technique
that made early FPS games possible — "Doom, upgraded," rendered entirely in the
browser with zero assets. It is intentionally **not** a modern 3D-engine title like
the *World War: Army Battle* screenshots: that fidelity needs a real engine
(Unity/Unreal), 3D models, and licensed art — an art-team effort. This is the
honest, runnable core: the shooting, the enemies, the weapons, the level. Nail the
feel here, then re-skin in an engine if you want to take it further.

## How the engine works (quick tour of `index.html`)

- **DDA raycaster** casts one ray per screen column to find the nearest wall,
  draws a vertical textured slice scaled by distance, and records that distance in
  a **z-buffer**.
- **Sprites** (enemies, pickups, blood) are billboarded, sorted far-to-near, and
  drawn column-by-column, skipping any column where the z-buffer says a wall is
  closer — that's what makes enemies hide behind corners.
- **Textures** are painted once onto small offscreen canvases at startup.
- **Sound** is synthesized live with the Web Audio API (no audio files).
