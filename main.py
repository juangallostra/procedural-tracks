import pygame, sys
from pygame.locals import *
import random as rn
import numpy as np
import scipy as sc
from scipy.spatial import ConvexHull

WIDTH = 400 
HEIGHT = 300

WHITE = [255, 255, 255]
RED = [255, 0, 0]

def random_points(min=10, max=20, margin=30):
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

def paint_point(surface, color, pos, radius=2):
    pygame.draw.circle(surface, color, pos, radius)
    # surface.fill(color, (pos, (1, 1)))

def paint_line(surface, color, init, end):
    pygame.draw.line(surface, color, init, end)

def main(debug=True):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    background_color = [58, 156, 53]
    screen.fill(background_color)

    # generate random points
    points = random_points(10, 20)
    hull = ConvexHull(points)
    if debug:
        draw_points(screen, WHITE, points)
        draw_convex_hull(hull, screen, points, RED)

    pygame.display.set_caption('Proc-Track')
    while True: # main game loop
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()

if __name__ == '__main__':
    main()
