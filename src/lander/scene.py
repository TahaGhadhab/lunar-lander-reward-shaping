"""A nicer renderer for LunarLander: textured moon, starfield, and a lander that looks like a rocket.

Purely cosmetic. It reads the physics state (body, legs, terrain, pad, exhaust particles) from the unwrapped
env and draws its own picture; it never changes the simulation. Callers still call `env.render()` so that
Gymnasium keeps its own particle bookkeeping exactly as in training and evaluation.
"""

from __future__ import annotations

import math
import os

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402
from pygame import gfxdraw  # noqa: E402

W, H, SCALE = 600, 400, 30.0  # same viewport and scale as gymnasium's LunarLander


def to_screen(x: float, y: float) -> tuple[float, float]:
    """World (metres, y up) -> screen pixels (y down)."""
    return x * SCALE, H - y * SCALE


def value_noise(shape: tuple[int, int], cell: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth random field in [0, 1]: random grid, smoothstep-interpolated."""
    h, w = shape
    grid = rng.random((h // cell + 2, w // cell + 2))
    ys, xs = np.mgrid[0:h, 0:w] / cell
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = ys - y0, xs - x0
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    top = grid[y0, x0] * (1 - fx) + grid[y0, x0 + 1] * fx
    bottom = grid[y0 + 1, x0] * (1 - fx) + grid[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def regolith_texture(rng: np.random.Generator) -> np.ndarray:
    """Grey lunar soil: multi-octave noise, fine grain, and craters with a dark floor and a lit rim."""
    n = sum(value_noise((H, W), c, rng) * a for c, a in ((64, 0.5), (24, 0.3), (8, 0.15), (3, 0.05)))
    shade = 0.55 + 0.35 * n + rng.normal(0, 0.035, (H, W))
    ys, xs = np.mgrid[0:H, 0:W]
    for _ in range(38):
        cx, cy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(4, 22)
        d = np.sqrt((xs - cx) ** 2 + ((ys - cy) / 0.55) ** 2) / r  # flattened: seen at a grazing angle
        floor = np.clip(1 - d, 0, 1) ** 0.6 * 0.22
        rim = np.exp(-((d - 1.0) ** 2) / 0.02) * 0.12
        shade += -floor + rim * np.where(ys < cy, 1.0, -0.6)  # rim lit from above, shadowed below
    shade = np.clip(shade, 0, 1)
    rgb = np.stack([shade * 205 + 25, shade * 200 + 25, shade * 192 + 28], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


class SceneRenderer:
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.stars = [(self.rng.uniform(0, W), self.rng.uniform(0, H * 0.85), self.rng.uniform(0.3, 1.0),
                       self.rng.uniform(0, 6.28), int(self.rng.choice([1, 1, 1, 2]))) for _ in range(170)]
        self.sky = self._sky()
        self.regolith = pygame.surfarray.make_surface(regolith_texture(self.rng).swapaxes(0, 1))
        self.metal = self._metal_texture()
        self._terrain_key = None
        self._ground = None

    # ---------- static layers
    def _sky(self) -> pygame.Surface:
        sky = pygame.Surface((W, H))
        for y in range(H):
            k = y / H
            sky.fill((int(4 + 10 * k), int(6 + 12 * k), int(18 + 30 * k)), (0, y, W, 1))
        glow = pygame.Surface((110, 110), pygame.SRCALPHA)
        for r in range(46, 34, -1):
            pygame.draw.circle(glow, (90, 150, 255, int(4 * (46 - r))), (55, 55), r)
        sky.blit(glow, (W - 125, 5))
        earth = pygame.Surface((90, 90), pygame.SRCALPHA)
        for r in range(34, 0, -1):  # shaded globe, lit from the left
            k = 1 - r / 34
            pygame.draw.circle(earth, (int(40 + 60 * k), int(90 + 80 * k), int(170 + 70 * k)),
                               (45 - int(8 * k), 45 - int(6 * k)), r)
        for cx, cy, rr in ((35, 38, 9), (52, 50, 7), (42, 58, 5)):  # continents
            pygame.draw.circle(earth, (70, 140, 80), (cx, cy), rr)
        night = pygame.Surface((90, 90), pygame.SRCALPHA)
        pygame.draw.circle(night, (0, 0, 10, 150), (62, 50), 34)
        earth.blit(night, (0, 0))
        disc = pygame.Surface((90, 90), pygame.SRCALPHA)
        pygame.draw.circle(disc, (255, 255, 255, 255), (45, 45), 34)
        earth.blit(disc, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        sky.blit(earth, (W - 115, 15))
        return sky

    def _metal_texture(self) -> pygame.Surface:
        """Brushed-metal fuselage texture: vertical grain and cylinder shading lit from the left."""
        w, h = 26, 70
        xs = np.linspace(-1, 1, w)
        cyl = 0.55 + 0.45 * np.cos(xs * 1.2) - 0.25 * np.clip(xs, 0, 1)
        grain = self.rng.normal(0, 0.04, (h, w)) + 0.05 * np.sin(np.arange(h)[:, None] * 0.9)
        v = np.clip(cyl[None, :] + grain, 0, 1)
        rgb = np.stack([v * 215 + 30, v * 218 + 30, v * 225 + 30], axis=-1).clip(0, 255).astype(np.uint8)
        return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    def _ground_layer(self, e) -> pygame.Surface:
        """Textured ground below the terrain line, cached until the terrain changes (new episode)."""
        polys = e.sky_polys
        key = tuple(round(c, 4) for p in polys for c in p[0]) + (round(e.helipad_y, 4),)
        if key == self._terrain_key:
            return self._ground
        top = [to_screen(*p[0]) for p in polys] + [to_screen(*polys[-1][1])]
        mask = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.polygon(mask, (255, 255, 255, 255), top + [(W, H), (0, H)])
        layer = pygame.Surface((W, H), pygame.SRCALPHA)
        layer.blit(self.regolith, (0, 0))
        shade = pygame.Surface((W, H), pygame.SRCALPHA)  # darker further below the surface
        tx, ty = [p[0] for p in top], [p[1] for p in top]
        for x in range(0, W, 2):
            sy = float(np.interp(x, tx, ty))
            for k in range(0, 120, 4):
                shade.fill((0, 0, 0, min(170, int(k * 1.4))), (x, int(sy + 18 + k), 2, 4))
            shade.fill((0, 0, 0, 170), (x, int(sy + 138), 2, H))
        layer.blit(shade, (0, 0))
        layer.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        pygame.draw.aalines(layer, (240, 240, 230), False, top)  # sunlit edge
        pygame.draw.lines(layer, (205, 205, 195), False, [(x, y + 1) for x, y in top], 1)
        self._pad(layer, e)
        self._terrain_key, self._ground = key, layer
        return layer

    def _pad(self, layer: pygame.Surface, e) -> None:
        x1, x2 = e.helipad_x1 * SCALE, e.helipad_x2 * SCALE
        y = H - e.helipad_y * SCALE
        pygame.draw.rect(layer, (70, 74, 82), (x1 - 4, y, x2 - x1 + 8, 7))
        pygame.draw.rect(layer, (120, 126, 136), (x1 - 4, y, x2 - x1 + 8, 2))
        for i, x in enumerate(range(int(x1), int(x2), 10)):  # hazard stripes
            pygame.draw.polygon(layer, (230, 190, 40) if i % 2 == 0 else (30, 30, 30),
                                [(x, y + 2), (x + 10, y + 2), (x + 6, y + 7), (x - 4, y + 7)])
        for leg_x in (x1 + 6, (x1 + x2) / 2, x2 - 6):
            pygame.draw.line(layer, (55, 58, 64), (leg_x, y + 7), (leg_x, y + 14), 3)
        for x in (x1, x2):  # masts with pennants; their lights blink in render()
            pygame.draw.line(layer, (190, 195, 205), (x, y), (x, y - 44), 2)
            pygame.draw.polygon(layer, (220, 70, 50), [(x, y - 44), (x, y - 34), (x + 18, y - 39)])

    # ---------- the lander
    def _rocket_sprite(self, action: int, t: int) -> pygame.Surface:
        """Drawn in body coordinates: sprite centre = body origin, nose up. 64 x 104 px."""
        s = pygame.Surface((64, 104), pygame.SRCALPHA)
        cx, cy = 32, 52

        def p(x, y):  # body px (y up) -> sprite px
            return int(cx + x), int(cy - y)

        flick = 0.75 + 0.25 * math.sin(t * 1.7) + 0.1 * math.sin(t * 5.3)
        if action == 2:  # main engine flame: outer orange, inner yellow, white core
            length = 30 * flick
            for col, wid, ln in (((255, 120, 30, 120), 9, length), ((255, 190, 60, 180), 6, length * 0.75),
                                 ((255, 250, 210, 235), 3, length * 0.45)):
                pygame.draw.polygon(s, col, [p(-wid, -15), p(wid, -15), p(0, -15 - ln)])
        for side, fire in ((-1, action == 1), (1, action == 3)):  # attitude thrusters, high on the hull
            if fire:
                for col, ln, wid in (((255, 160, 60, 160), 13 * flick, 3), ((255, 240, 200, 230), 7, 1.5)):
                    pygame.draw.polygon(s, col, [p(side * 11, 16 + wid), p(side * 11, 16 - wid),
                                                 p(side * (11 + ln), 16)])
        for side in (-1, 1):  # fins
            fin = [p(side * 9, 2), p(side * 18, -8), p(side * 18, -12), p(side * 9, -9)]
            pygame.draw.polygon(s, (175, 45, 40), fin)
            pygame.draw.aalines(s, (110, 25, 25), True, fin)
        pygame.draw.polygon(s, (60, 62, 70), [p(-5, -9), p(5, -9), p(7, -16), p(-7, -16)])  # engine bell
        pygame.draw.polygon(s, (100, 103, 113), [p(-5, -9), p(-1, -9), p(-3, -16), p(-7, -16)])
        if action == 2:
            pygame.draw.ellipse(s, (255, 200, 120), (cx - 6, cy + 14, 12, 4))
        body = pygame.Surface((26, 70), pygame.SRCALPHA)  # textured fuselage clipped to a rounded body
        body.blit(self.metal, (0, 0))
        clip = pygame.Surface((26, 70), pygame.SRCALPHA)
        pygame.draw.rect(clip, (255, 255, 255, 255), (2, 24, 22, 37), border_radius=4)
        body.blit(clip, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        s.blit(body, (cx - 13, cy - 52))  # cylinder spans body y -9..+28
        nose = [p(-11, 21), p(11, 21), p(0, 46)]
        pygame.draw.polygon(s, (190, 50, 45), nose)
        pygame.draw.polygon(s, (228, 88, 72), [p(-11, 21), p(-2, 21), p(0, 46)])
        pygame.draw.line(s, (90, 90, 100), p(-11, 21), p(11, 21), 2)
        for y in (9, -2):  # panel seams and rivets
            pygame.draw.line(s, (115, 119, 130), p(-11, y), p(11, y), 1)
            for x in range(-9, 10, 4):
                s.set_at(p(x, y + 2), (85, 87, 96))
        pygame.draw.line(s, (150, 154, 165), p(5, 20), p(5, -8), 1)
        gfxdraw.filled_circle(s, *p(0, 14), 5, (40, 45, 55))  # porthole
        gfxdraw.filled_circle(s, *p(0, 14), 4, (60, 130, 200))
        gfxdraw.filled_circle(s, *p(-1, 15), 2, (190, 225, 255))
        gfxdraw.aacircle(s, *p(0, 14), 5, (160, 165, 175))
        pygame.draw.aalines(s, (45, 48, 58), False, [p(-11, -9), p(-11, 21), p(0, 46), p(11, 21), p(11, -9)])
        return s

    def _legs(self, surf: pygame.Surface, e) -> None:
        for leg, side in zip(e.legs, (-1, 1), strict=True):
            top = to_screen(*leg.GetWorldPoint((0, 8 / SCALE)))
            foot = to_screen(*leg.GetWorldPoint((0, -8 / SCALE)))
            hip = to_screen(*e.lander.GetWorldPoint((side * 9 / SCALE, -2 / SCALE)))
            knee = to_screen(*e.lander.GetWorldPoint((side * 11 / SCALE, -9 / SCALE)))
            pygame.draw.line(surf, (150, 155, 165), hip, top, 3)
            pygame.draw.line(surf, (110, 115, 125), knee, foot, 2)
            pygame.draw.line(surf, (205, 210, 220), top, foot, 3)
            dx, dy = 6 * math.cos(leg.angle), -6 * math.sin(leg.angle)
            pad = [(foot[0] - dx, foot[1] - dy), (foot[0] + dx, foot[1] + dy),
                   (foot[0] + dx * 0.7, foot[1] + dy + 3), (foot[0] - dx * 0.7, foot[1] - dy + 3)]
            pygame.draw.polygon(surf, (240, 200, 80) if leg.ground_contact else (170, 175, 185), pad)

    def _particles(self, surf: pygame.Surface, e) -> None:
        glow = pygame.Surface((W, H), pygame.SRCALPHA)
        for obj in e.particles:
            ttl = max(0.0, min(1.0, obj.ttl))
            x, y = to_screen(*obj.position)
            r = 2 + 4 * (1 - ttl)
            gfxdraw.filled_circle(glow, int(x), int(y), int(r + 2), (150, 150, 160, int(90 * ttl)))  # smoke
            gfxdraw.filled_circle(glow, int(x), int(y), int(r),
                                  (255, int(140 + 100 * ttl), int(60 * ttl), int(200 * ttl + 30)))  # ember
        surf.blit(glow, (0, 0))

    def _explosion(self, surf: pygame.Surface, pos, k: int) -> None:
        x, y = pos
        rng = np.random.default_rng(k // 2)
        for i, col in enumerate(((200, 60, 30), (255, 140, 40), (255, 230, 150))):
            r = max(1, int(6 + k * (2.2 - i * 0.6)))
            gfxdraw.filled_circle(surf, int(x), int(y), r, (*col, max(0, 210 - k * 5)))
        for _ in range(16):
            ang, dist = rng.uniform(0, math.pi), k * rng.uniform(1.5, 4)
            gfxdraw.filled_circle(surf, int(x + dist * math.cos(ang)), int(y - dist * math.sin(ang)), 2,
                                  (255, 200, 120, max(0, 230 - k * 6)))

    # ---------- public
    def render(self, env, action: int = 0, t: int = 0, crash_frame: int | None = None) -> np.ndarray:
        """RGB frame (400, 600, 3). `crash_frame` = k-th frame of an explosion replacing the lander."""
        e = env.unwrapped
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        surf.blit(self.sky, (0, 0))
        for x, y, b, ph, size in self.stars:
            v = int(255 * b * (0.75 + 0.25 * math.sin(t * 0.08 + ph)))
            if size == 1:
                surf.set_at((int(x), int(y)), (v, v, v))
            else:
                gfxdraw.filled_circle(surf, int(x), int(y), 1, (v, v, min(255, v + 20)))
        surf.blit(self._ground_layer(e), (0, 0))
        for i, x in enumerate((e.helipad_x1 * SCALE, e.helipad_x2 * SCALE)):  # blinking mast lights
            on = (t // 25 + i) % 2 == 0
            yy = int(H - e.helipad_y * SCALE - 46)
            gfxdraw.filled_circle(surf, int(x), yy, 3, (255, 80, 60) if on else (90, 30, 25))
            if on:
                gfxdraw.filled_circle(surf, int(x), yy, 7, (255, 80, 60, 55))
        self._particles(surf, e)
        if crash_frame is None:
            self._legs(surf, e)
            sprite = pygame.transform.rotate(self._rocket_sprite(action, t), math.degrees(e.lander.angle))
            surf.blit(sprite, sprite.get_rect(center=to_screen(*e.lander.position)))
        else:
            self._explosion(surf, to_screen(*e.lander.position), crash_frame)
        return pygame.surfarray.array3d(surf).swapaxes(0, 1).copy()
