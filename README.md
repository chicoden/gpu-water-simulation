# GPU Water Simulation
Learning the WebGPU API and how to simulate water at the same time .-. \
Spent a disproportionate amount of time figuring out how to optimally index vertices
in the marching cubes pass, it's possible but I decided to stop wasting time and move
on since my hybrid approach of only reusing vertices within cells was working just fine,
honestly could have just stuck with my first version that didn't even bother with
indexing. I then spent another disproportionate amount of time wondering why my free
camera kept unintentionally rolling and actually it makes sense if you think about
how the yaw and pitch rotations accumulate. I tried keeping a fixed up vector to yaw around
that only changes when explicitly rolling but naturally, it results in yawing looking like
rolling when looking along it so I gave up on that and embraced the roll. So now I just
need to implement the fluid simulation. And I'm not sure marching cubes is going to
be the best solution for rendering it after watching Simon Lague's Coding Adventure on fluid
rendering, but we'll see.

# Resources
- [Polygonising a scalar field](http://www.paulbourke.net/geometry/polygonise/)
- [17 - How to write an Eulerian fluid simulator with 200 lines of code.](https://youtu.be/iKAVRgIrUOU)
- [WebGPU Fundamentals](https://webgpufundamentals.org/)