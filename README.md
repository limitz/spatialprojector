# Spatial Transformer / Spatial Projector

Based initially on this tutorial: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html, 
which is based on this paper: https://arxiv.org/abs/1506.02025

I implemented a Spatial Projector, a drop in replacement for Spatial Transformer, that learns a perspective transform instead of an affine transform.
I did this by adding a `perspective_grid` function that is a 3x3 variant of the `affine_grid` function in ATen.

I also added 2 examples, the first one is the one from the tutorial linked to above, the second one learns to rectify an image that is assumed to have 
a rectangle annotated with a 4 point polygon.
