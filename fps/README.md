# Breach Protocol — a raycasting first-person shooter

A single-file, no-dependency browser FPS built on the same engine lineage as the
original Doom: **raycasting**. Real 3D perspective, textured walls, sprite
enemies, and a **Doom-derived bestiary and arsenal** rescaled for a modern,
"Doom-upgraded" feel. Everything — textures, monsters, weapons, sound — is
generated procedurally in code. No images, no audio files, no build step.

## Play it

Open `fps/index.html` in any modern browser.

```
open fps/index.html        # macOS
xdg-open fps/index.html    # Linux
start fps\index.html       # Windows
```

## Controls

**Desktop**
| Action        | Input                                            |
|---------------|--------------------------------------------------|
| Move          | `W` `A` `S` `D` / arrow keys                     |
| Look          | Mouse (click screen to lock; `Esc` releases)     |
| Fire          | Left mouse (hold for automatic weapons)          |
| Switch weapon | `1`–`7`, or mouse **scroll wheel**               |

**Touch:** left stick to move, drag the right side to look, **FIRE** button,
**SWAP** to cycle weapons.

## The bestiary (rescaled from classic Doom)

| Monster       | HP   | Attack            | Behavior                                  |
|---------------|------|-------------------|-------------------------------------------|
| Zombieman     | 25   | Hitscan           | Weak, instant-hit chaff                    |
| Shotgun Guy   | 40   | Hitscan spread    | Dangerous at mid-range                     |
| Imp           | 60   | Fireball + claw   | Lobs dodgeable projectiles                 |
| Pinky         | 150  | Melee             | Fast charger, no ranged attack             |
| Cacodemon     | 400  | Plasma spit       | Floating bullet-sponge                     |
| Baron of Hell | 1000 | Green plasma+claw | Mini-boss                                  |

Two signature Doom mechanics are implemented:

- **Pain-state stun-lock** — every hit has a monster-specific chance to flinch
  the target, interrupting its attack. Rapid-fire weapons (chaingun/plasma) can
  stun-lock weaker demons to death. Barons barely flinch.
- **Monster infighting** — when a demon's stray projectile hits another demon,
  the victim turns on the shooter. Bait them into crossfire.

## The arsenal (Doom weapons, upgraded)

| # | Weapon        | Ammo    | Notes                                        |
|---|---------------|---------|----------------------------------------------|
| 1 | Pistol        | ∞       | Reliable sidearm                             |
| 2 | Shotgun       | shells  | 7-pellet hitscan spread                      |
| 3 | Super Shotgun | shells  | 20 pellets, brutal up close (2 shells/shot)  |
| 4 | Chaingun      | bullets | Fast full-auto, great for stun-locking       |
| 5 | Rocket        | rockets | **Splash damage** — mind your own feet       |
| 6 | Plasma        | cells   | Fast full-auto energy bolts                  |
| 7 | BFG 9000      | cells   | 40 cells, huge blast                         |

Ammo is tracked by type (**bullets / shells / rockets / cells**) and picked up
from demon drops, along with health and armor. Start with pistol, shotgun and
chaingun; find the rest as weapon drops. **Armor absorbs half** of incoming
damage until it runs out.

## What this is (and isn't)

A genuine, playable FPS using the **raycasting** technique that made early
first-person shooters possible — "Doom, upgraded," in the browser with zero
assets. It is intentionally **not** a modern 3D-engine title like the
*World War: Army Battle* screenshots: that fidelity needs a real engine
(Unity/Unreal), 3D models, and licensed art. This is the honest core — the
demons, weapons, and feel are real and tunable now.

## Engine tour (`index.html`)

- **DDA raycaster**: one ray per screen column finds the nearest wall, draws a
  vertical textured slice, and stores its distance in a **z-buffer**.
- **Billboard sprites** (demons, projectiles, pickups, blood) are sorted
  far-to-near and drawn column-by-column, skipping any column a wall is closer
  in — that's what hides demons behind corners.
- **Projectiles** are real world entities with sub-stepped movement, wall
  collision, splash radius, and infighting checks.
- **Sound** is synthesized live with the Web Audio API.
