# procedural-tracks
Procedural race track generation with Python.

Following the article found [here](https://www.gamasutra.com/blogs/GustavoMaciel/20131229/207833/Generating_Procedural_Racetracks.php).

The tracks obtained by my implementation are far from ideal, but it might be a good starting point for further work.

## Process

As explained in the post mentioned above, which I strongly recommend you to read, below you can find the steps taken to generate a track. Note that there are some restrictions and parameters used in the code which are not explained in the summary presented below. You can dive in the code to learn more about them.

The outline of the algorithm is:
1. Generate a set of random points (white points)
2. Compute the points that, from the set of all points generated in step 1, form the convex Hull (red lines)
3. For each pair of consecutive points in the convex hull, compute the midpoint and displace it by a random bounded amount
4. Push points whose distance is less than a predefined threshold apart an limit the max angle between them.
5. From the final set of points, compute a spline that passess through all of them.

## Example layouts

By following this steps we can get the layout of the track.

![Example Tracks](/img/tracks.png)

## Example racetracks

Once the layout has been obtained we can draw the racetrack to get a more appealing result. 

![Example drawn tracks](/img/tracks_drawn.png)

## Example racetracks

I have also tried to, given a minimum and maximum track angle corners, draw kerbs on the track. This hasn't been really successful. Anyway, below you can see some results of the resulting tracks after adding the kerbs.

![Example tracks with kerbs](/img/tracks_kerb.png)

## TODO

- [ ] Fix inconsistencies in data structures (numpy arrays, lists, tuples, etc.).
- [ ] Refactor code when track drawing is finished.
- [ ] Test other interpolation methods to obtain the smoothed track.
- [ ] Tune parameters.
- [x] Add kerbs to corners. 