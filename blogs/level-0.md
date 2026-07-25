# Level 0

## concepts as vectors(semantic spaces)

In this part we illustrate how Artificial-Neural-Networks(ANNs) encode their understanding of different concepts & relationships.

lets start w/ a simple example:

```python

house_cat = [0.2, 0.9]  # Small, highly domesticated
tiger     = [0.8, 0.1]  # Medium-large, highly wild
wolf      = [0.6, 0.15] # Medium, wild
chihuahua = [0.1, 0.95] # Very small, highly domesticated

```

in a more visual form:

![2D-animal-vector-space](../media/2D-animal-vector-space.png)


**The main idea is this:**\
you can describe concepts(tigers, capt &...) which may contain many features and traits as a vector, a point in a multi-dimensional space. how far along each dimension the point is(tiger=0.8 along the size dim, cat only 0.2) will indicate the intensity of that trait. 


adding more dimensions to our space, allows us to describe concepts in terms of more features, enabling more accurate representation & understanding.
![3D-animal-vector-space](../media/3D-animal-vector-space.png)


lets see a few examples in language & images that where learned by ANNs:

![]()
![]()
![]()

cool analogy:
playing 20-questions is like doing binary search(yes or no, not how much, big or small, not how big) over the space of all concepts!

## the linear layer & batched IO (XW=Y)
## gradient descent via back-propagation
## normalization & optimizers
