import numpy as np
import math
from PIL import Image, ImageFilter, ImageChops
from numba import njit
import multiprocessing as mp
from tqdm import tqdm
import random
import os

# Bildeinstellungen
RESOLUTION_X = 1920
RESOLUTION_Y = 1080
FILENAME = "black_hole_with_stars_fast.png"

# Simulationseinstellungen
TIMESTEP = 0.1
SCHWARZSCHILD_RADIUS = 50.0
MAX_STEPS = 5000
MAX_DISTANCE = 1000

# Farben
COLOR_BLACK = np.array([0, 0, 0], dtype=np.uint8)
COLOR_BACKGROUND = np.array([5, 5, 10], dtype=np.uint8)

# Akkretionsscheiben-Parameter
DISK_INNER_RADIUS = SCHWARZSCHILD_RADIUS * 1.5
DISK_OUTER_RADIUS = SCHWARZSCHILD_RADIUS * 6.0
DISK_COLOR_INNER = np.array([220.0, 235.0, 255.0])
DISK_COLOR_OUTER = np.array([0.0, 20.0, 100.0])

FADE_ZONE_WIDTH = 0.25

# Parameter für die Textur
NOISE_SCALE = 80.0
NOISE_STRENGTH = 0.6

# Sternenkarten-Einstellungen
STAR_MAP_FILENAME = "star_map.png"
STAR_MAP_WIDTH = 4096
STAR_MAP_HEIGHT = 2048
STAR_COUNT = 15000


def create_star_map():
    if os.path.exists(STAR_MAP_FILENAME):
        print(f"'{STAR_MAP_FILENAME}' existiert bereits, Erstellung wird übersprungen.")
        return
    print(f"Erstelle Sternenkarte '{STAR_MAP_FILENAME}'...")
    img = Image.new('RGB', (STAR_MAP_WIDTH, STAR_MAP_HEIGHT), color=(2, 2, 5))
    pixels = img.load()
    for _ in range(STAR_COUNT):
        x, y = random.randint(0, STAR_MAP_WIDTH - 1), random.randint(0, STAR_MAP_HEIGHT - 1)
        brightness = random.randint(150, 255)
        star_color = (
        int(brightness * random.uniform(0.9, 1.0)), int(brightness * random.uniform(0.9, 1.0)), brightness)
        pixels[x, y] = star_color
        if random.random() < 0.1:
            size = random.randint(1, 2)
            for dx in range(-size, size + 1):
                for dy in range(-size, size + 1):
                    if dx == 0 and dy == 0: continue
                    dist_sq = dx * dx + dy * dy
                    if dist_sq <= size * size:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < STAR_MAP_WIDTH and 0 <= ny < STAR_MAP_HEIGHT:
                            falloff = 1.0 - math.sqrt(dist_sq) / (size + 1)
                            glow_color = tuple(int(c * falloff * 0.5) for c in star_color)
                            current_color = pixels[nx, ny]
                            pixels[nx, ny] = tuple(min(255, current_color[i] + glow_color[i]) for i in range(3))
    img.save(STAR_MAP_FILENAME)
    print("Sternenkarte erstellt.")


@njit
def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)


@njit
def lerp(t, a, b): return a + t * (b - a)


@njit
def grad(hash, x, y):
    h = hash & 15
    u, v = (x, y) if h < 8 else (y, x)
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


@njit
def perlin_noise(x, y, p_in):
    p = np.array(p_in)
    X, Y = int(math.floor(x)) & 255, int(math.floor(y)) & 255
    x, y = x - math.floor(x), y - math.floor(y)
    u, v = fade(x), fade(y)
    A, B = p[X] + Y, p[X + 1] + Y
    g1, g2 = grad(p[A], x, y), grad(p[B], x - 1, y)
    g3, g4 = grad(p[A + 1], x, y - 1), grad(p[B + 1], x - 1, y - 1)
    return lerp(v, lerp(u, g1, g2), lerp(u, g3, g4))


p_table = list(range(256));
random.shuffle(p_table);
p_table.extend(p_table)


@njit
def get_accretion_disk_color(position, p_noise_table):
    radius = np.linalg.norm(position[0:2])
    t = max(0.0, min(1.0, 1.0 - (radius - DISK_INNER_RADIUS) / (DISK_OUTER_RADIUS - DISK_INNER_RADIUS)))
    base_color = lerp(t ** 2, DISK_COLOR_OUTER, DISK_COLOR_INNER)
    noise_val = perlin_noise(position[0] / NOISE_SCALE, position[1] / NOISE_SCALE, p_noise_table)
    brightness_noise = 1.0 + noise_val * NOISE_STRENGTH
    brightness_doppler = 1.5 + (position[1] / radius) * 1.2
    color_before_fade = base_color * brightness_noise * brightness_doppler
    fade_alpha = t / FADE_ZONE_WIDTH if t < FADE_ZONE_WIDTH else 1.0
    background_float = COLOR_BACKGROUND.astype(np.float64)
    final_color = lerp(fade_alpha, background_float, color_before_fade)
    return np.clip(final_color, 0, 255).astype(np.uint8)


@njit
def get_star_color(direction_vector, star_map):
    norm = np.linalg.norm(direction_vector)
    if norm == 0: return COLOR_BACKGROUND
    vec = direction_vector / norm
    phi, theta = math.atan2(vec[1], vec[0]), math.acos(vec[2])
    u, v = (phi + math.pi) / (2 * math.pi), theta / math.pi
    x = min(star_map.shape[1] - 1, int(u * star_map.shape[1]))
    y = min(star_map.shape[0] - 1, int(v * star_map.shape[0]))
    return star_map[y, x]


@njit
def trace_photon(start_pos, start_vel, p_noise_table, star_map):
    rel_pos, r = start_pos.copy(), np.linalg.norm(start_pos)
    L_vec = np.cross(rel_pos, start_vel)
    L = np.linalg.norm(L_vec)
    pr = np.dot(rel_pos, start_vel) / r
    if L < 1e-9: return get_star_color(start_vel, star_map)
    u_vec, w_vec = rel_pos / r, L_vec / L
    v_vec = np.cross(w_vec, u_vec)
    phi, current_rel_pos = 0.0, rel_pos
    for _ in range(MAX_STEPS):
        last_rel_pos = current_rel_pos
        if r < SCHWARZSCHILD_RADIUS: return COLOR_BLACK
        if r > MAX_DISTANCE:
            final_direction = current_rel_pos / np.linalg.norm(current_rel_pos)
            return get_star_color(final_direction, star_map)
        adaptive_timestep = max(0.01, min(0.2, TIMESTEP * (r / (3 * SCHWARZSCHILD_RADIUS))))
        dr, dphi = pr, L / r ** 2
        dpr = (L ** 2 / r ** 3) - (1.5 * SCHWARZSCHILD_RADIUS * L ** 2 / r ** 4)
        r, phi, pr = r + dr * adaptive_timestep, phi + dphi * adaptive_timestep, pr + dpr * adaptive_timestep
        current_rel_pos = r * (np.cos(phi) * u_vec + np.sin(phi) * v_vec)
        if last_rel_pos[2] * current_rel_pos[2] < 0:
            t_intersect = -last_rel_pos[2] / (current_rel_pos[2] - last_rel_pos[2])
            intersection_point = last_rel_pos + t_intersect * (current_rel_pos - last_rel_pos)
            radius_at_intersection = np.linalg.norm(intersection_point[0:2])
            if DISK_INNER_RADIUS <= radius_at_intersection <= DISK_OUTER_RADIUS:
                return get_accretion_disk_color(intersection_point, p_noise_table)
    final_direction = current_rel_pos / np.linalg.norm(current_rel_pos)
    return get_star_color(final_direction, star_map)


# OPTIMIERUNG: Initializer-Funktion für jeden Worker-Prozess
# Diese Funktion wird EINMAL pro Prozess aufgerufen.
def init_worker(star_map_data, p_noise_table_data):
    # Speichere die Daten in globalen Variablen, die nur für diesen Prozess sichtbar sind
    global worker_star_map
    global worker_p_table
    worker_star_map = star_map_data
    worker_p_table = p_noise_table_data
    print(f"Worker-Prozess {os.getpid()} initialisiert.")


# OPTIMIERUNG: Die Task-Funktion nimmt die großen Daten nicht mehr als Argument entgegen
def trace_pixel_task(args):
    i, j = args  # Nur noch Pixelkoordinaten

    # Greife auf die prozess-globalen Variablen zu
    global worker_star_map
    global worker_p_table

    camera_pos = np.array([-400.0, 0.0, 30.0])
    screen_center = np.array([-350.0, 0.0, 30.0])
    screen_width = 250.0
    screen_height = (screen_width / RESOLUTION_X) * RESOLUTION_Y
    x_offset = (i / RESOLUTION_X - 0.5) * screen_width
    y_offset = (j / RESOLUTION_Y - 0.5) * screen_height
    pixel_pos = screen_center + np.array([0, x_offset, -y_offset])
    start_velocity = pixel_pos - camera_pos
    start_velocity /= np.linalg.norm(start_velocity)
    max_impact_parameter = DISK_OUTER_RADIUS * 2.0
    impact_parameter = np.linalg.norm(np.cross(camera_pos, start_velocity))

    if impact_parameter < max_impact_parameter:
        color = trace_photon(camera_pos, start_velocity, worker_p_table, worker_star_map)
    else:
        color = get_star_color(start_velocity, worker_star_map)
    return i, j, color


def main():
    create_star_map()
    print(f"Lade Sternenkarte '{STAR_MAP_FILENAME}'...")
    star_map_img = Image.open(STAR_MAP_FILENAME)
    star_map_data = np.array(star_map_img, dtype=np.uint8)
    p_table_tuple = tuple(p_table)
    print("Sternenkarte geladen.")

    # OPTIMIERUNG: Die Task-Liste enthält jetzt nur noch die Pixelkoordinaten.
    tasks = [(i, j) for i in range(RESOLUTION_X) for j in range(RESOLUTION_Y)]
    final_pixels = np.zeros((RESOLUTION_Y, RESOLUTION_X, 3), dtype=np.uint8)

    print(f"Starte Raytracing mit {mp.cpu_count()} Kernen...")

    # OPTIMIERUNG: Pool wird mit der init_worker Funktion und den Daten gestartet.
    with mp.Pool(initializer=init_worker, initargs=(star_map_data, p_table_tuple)) as pool:
        # OPTIMIERUNG: imap_unordered ist oft schneller, chunksize reduziert den Overhead.
        # Eine gute Chunksize-Heuristik ist, die Gesamtmenge der Tasks in mehrere
        # sinnvolle Pakete pro CPU aufzuteilen.
        num_tasks = len(tasks)
        chunksize = max(1, (num_tasks // (mp.cpu_count() * 4)))

        results = list(tqdm(pool.imap_unordered(trace_pixel_task, tasks, chunksize=chunksize), total=num_tasks))

    for i, j, color in results:
        final_pixels[j, i] = color

    base_img = Image.fromarray(final_pixels, 'RGB')
    brightness_mask = base_img.point(lambda p: 255 if p > 200 else 0).convert('L')
    glow_map = brightness_mask.filter(ImageFilter.GaussianBlur(radius=25))
    final_img = ImageChops.screen(base_img, glow_map.convert('RGB'))
    print(f"\nRaytracing abgeschlossen. Bild wird als '{FILENAME}' gespeichert.")
    final_img.save(FILENAME)
    final_img.show()


if __name__ == "__main__":
    # Numba JIT-Kompilierung "aufwärmen"
    print("Kompiliere Numba-Funktionen...")
    dummy_star_map = np.zeros((16, 16, 3), dtype=np.uint8)
    trace_photon(np.array([-400.0, 10.0, 30.0]), np.array([1.0, 0.0, 0.0]), tuple(p_table), dummy_star_map)
    print("Kompilierung abgeschlossen.")
    main()