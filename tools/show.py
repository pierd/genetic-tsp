#!/usr/bin/env python3

import sys
import re

CITY_PATTERN = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s*$')
SIZE = 800
RADIUS = 2
LINE_WIDTH = 1
PADDING = 10

if __name__ == '__main__':
    solved_cities = []
    cost = None
    try:
        with open(sys.argv[2]) as solution:
            parts = solution.readline().split()
            cost = int(parts.pop()) # remove last element (length)
            for part in parts:
                solved_cities.append(int(part))
    except:
        pass
    cities = {}
    with open(sys.argv[1]) as f:
        for line in f:
            match = CITY_PATTERN.match(line)
            if match:
                city = int(match.group(1))
                city_x = int(match.group(2))
                city_y = int(match.group(3))
                cities[city] = (city_x, city_y)

    min_x = min(cities.values(), key=lambda c: c[0])[0]
    max_x = max(cities.values(), key=lambda c: c[0])[0]
    min_y = min(cities.values(), key=lambda c: c[1])[1]
    max_y = max(cities.values(), key=lambda c: c[1])[1]
    width = max_x - min_x
    height = max_y - min_y
    ratio = max(width, height) / SIZE
    transform_x = lambda x: (x - min_x) / ratio + PADDING
    transform_y = lambda y: (y - min_y) / ratio + PADDING
    print(f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE + PADDING * 2}" height="{SIZE + PADDING * 2}">')
    if cost is not None:
        print(f'<text x="{PADDING * 2}" y="{PADDING * 2}" class="small">{cost}</text>')
    
    for (city_x, city_y) in cities.values():
        print(f'<circle cx="{transform_x(city_x)}" cy="{transform_y(city_y)}" r="{RADIUS}" fill="red" />')
    for i in range(len(solved_cities)):
        city1 = solved_cities[i - 1]    # -1 on first iteration (connect last city to first)
        city2 = solved_cities[i]
        city1_x, city1_y = cities[city1]
        city2_x, city2_y = cities[city2]
        print(f'<line x1="{transform_x(city1_x)}" y1="{transform_y(city1_y)}" x2="{transform_x(city2_x)}" y2="{transform_y(city2_y)}" stroke="blue" stroke-width="{LINE_WIDTH}" />')
    print('</svg>')
