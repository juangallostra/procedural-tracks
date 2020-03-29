import pygame, sys
from pygame.locals import *
import random as rn
import math
import numpy as np
import scipy as sc
from scipy.spatial import ConvexHull

WIDTH = 400 
HEIGHT = 300

WHITE = [255, 255, 255]
RED = [255, 0, 0]
BLUE = [0, 0, 255]
GRASS_GREEN = [58, 156, 53]

def random_points(min=10, max=20, margin=25):
    pointCount = rn.randrange(min, max+1, 1)
    points = []
    for i in range(pointCount):
        x = rn.randrange(margin, WIDTH - margin + 1, 1)
        y = rn.randrange(margin, HEIGHT -margin + 1, 1)
        points.append((x, y))
    return np.array(points)

def draw_points(surface, color, points):
    for p in points:
        paint_point(surface, color, p)

def get_track_points(hull, points):
    # get the original points from the random 
    # set that will be used as the track starting shape
    return np.array([points[hull.vertices[i]] for i in range(len(hull.vertices))])


def make_rand_vector(dims):
    vec = [rn.gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def shape_track(track_points, difficulty=0.6, max_displacement=45):
    track_set = [[0,0] for i in range(len(track_points)*2)] 
    for i in range(len(track_points)):
        displacement = math.pow(rn.random(), difficulty) * max_displacement
        disp = [displacement * i for i in make_rand_vector(2)]
        track_set[i*2] = track_points[i]
        track_set[i*2 + 1][0] = int((track_points[i][0] + track_points[(i+1)%len(track_points)][0]) / 2 + disp[0])
        track_set[i*2 + 1][1] = int((track_points[i][1] + track_points[(i+1)%len(track_points)][1]) / 2 + disp[1])
    return track_set

def draw_convex_hull(hull, surface, points, color):
    for i in range(len(hull.vertices)-1):
        paint_line(surface, color, points[hull.vertices[i]], points[hull.vertices[i+1]])
        # close the polygon
        if i == len(hull.vertices) - 2:
            paint_line(
                surface,
                color,
                points[hull.vertices[0]],
                points[hull.vertices[-1]]
            )
def draw_lines_from_points(surface, color, points):
    for i in range(len(points)-1):
        paint_line(surface, color, points[i], points[i+1])
        # close the polygon
        if i == len(points) - 2:
            paint_line(
                surface,
                color,
                points[0],
                points[-1]
            )

def paint_point(surface, color, pos, radius=2):
    pygame.draw.circle(surface, color, pos, radius)
    # surface.fill(color, (pos, (1, 1)))

def paint_line(surface, color, init, end):
    pygame.draw.line(surface, color, init, end)

def main(debug=True):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    background_color = GRASS_GREEN
    screen.fill(background_color)

    # generate random points
    points = random_points(10, 20)
    hull = ConvexHull(points) # points may have to be pushed away
    track_points = shape_track(get_track_points(hull, points))
    if debug:
        draw_points(screen, WHITE, points)
        draw_convex_hull(hull, screen, points, RED)
        draw_points(screen, BLUE, track_points)
        draw_lines_from_points(screen, BLUE, track_points)

    pygame.display.set_caption('Procedural Race Track')
    while True: # main game loop
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()

if __name__ == '__main__':
    main()
