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

## Graphics — a full software renderer

The world is drawn a pixel at a time into a framebuffer, not with flat colored
strips. That buys real fidelity:

- **Perspective-correct floor AND ceiling casting** — every floor/ceiling pixel
  samples a 128px texture (metal grating, panelled ceiling), not a gradient.
- **Per-pixel wall texturing** with 128px detailed textures (bevelled panels,
  rivets, hazard stripes, tech screens, rusted structure, a glowing core).
- **Distance fog + dynamic lighting** — light falls off with depth, the reactor
  is self-lit, and firing throws a real **muzzle-flash light** onto nearby walls,
  floor and demons.
- **Animated, grounded sprites** — demons have walk cycles, attack poses and
  death animations, are shaded into the scene's fog, and their feet sit correctly
  on the floor at any distance (a per-column z-buffer hides them behind corners).
- **Movement that feels right** — momentum/acceleration, head-bob, weapon sway &
  inertia, and strafe tilt, so the camera and gun move the way a body would.
- **Post-processing** — additive **bloom** on emitters and muzzle flash, a
  **vignette**, film **grain**, ejecting shell casings, sparks, smoke and blood.
- **Sound** synthesized live with the Web Audio API.

## What this is (and isn't)

A genuine, playable FPS using the **raycasting** technique that made early
first-person shooters possible — "Doom, upgraded," in the browser with zero
assets, now with a per-pixel software renderer and a post-FX stack. It is
intentionally **not** a modern 3D-engine title like the *World War: Army Battle*
screenshots: that photoreal fidelity needs a real engine (Unity/Unreal), 3D
models, and licensed art — an art-team effort. This is the honest, high-effort
ceiling for a single self-contained file.
