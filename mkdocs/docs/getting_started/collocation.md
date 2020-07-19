# What is collocation?

Dymos is focused on direct optimization methods based on an implicit integration technique, collocation.

The _state_ of a dynamic system defines its configuration at a point in time.
Common state variables for dynamic systems are position, velocity, and mass.
Given an initial state, a time duration, and ordinary differential equations that define the evolution of the states in time, the state of the system at the end of the time duration can be determined.
This is known as the _initial value problem_ (IVP) and there are numerous techniques and software packages that can be used to solve the IVP.
The initial value problem is typically the basis for the simulation of systems.
This time-marching approach is the basis for _shooting methods_ in optimization.
While Dymos supports shooting methods, its focus is on implicit collocation techniques, for performance reasons.

First, a few assumptions about the dynamic system:

1.  Over portions of the trajectory, the state is continuous (there are no instantaneous changes in the states)
2.  Over portions of the trajectory, the state is continuous in its first derivative w.r.t. time. (there are no instantaneous changes in the state rates)

If those two conditions are met, then it's reasonable fit a polynomial to a plot of the state value over time.
Given the initial or final value of a state, and values of its rate at some number of points across the trajectory, then a polynomial can be fit whose initial or final value matches, and whose rates match the sampled rates.
The value of the state at any point throughout the trajectory can then be inferred by interpolating the collocated polynomial.

Solving this trajectory implicitly basically means _guessing_ values for the states along the trajectory.
While one could use a power series polynomial coefficients ($a t^2 + b t + c), these coefficients can have vastly different values depending on the shape of the curve, making them difficult to scale numerically.
Instead, Lagrange interpolating polynomials are typically used, where the actual values to be interpolated at various points are the implicit variables.
This makes scaling the problem easier, and gives the implicit variables more intuitive meaning, making it easier to provide an initial guess.

## Example:  Simulating the fall of an object

As a "hello, world" problem for Dymos, consider the vertical fall of an object acting under gravity - like Galileo dropping cannonballs from the Tower of Pisa.
Ignoring aerodynamics and treating the ball as a point mass in a rectilinear gravity field, the motion is governed by:

\begin{align}
    \ddot{y} = -g
\end{align}

The techniques in Dymos are generally centered around first-order dynamics, so we convert the above equation into the following system of ODEs.

\begin{align}
    \dot{y} &= v_y \\
    \dot{v_y} &= -g
\end{align}

One common way to propagate this trajectory would be to use an "explicit time-marching" approach.
That is, we can use Euler's method or a Runge-Kutta approach to propagate the trajectory of the ball, one small slice of time at a time.
When performing trajectory optimization, the trajectory is typically evaluated at least dozens of time.
If the ODE for the system is computationally expensive, this time-marching approach can take a very long time.

Instead, we can use an implicit simulation technique to more efficiently model the trajectory.
First, let's assume that the value of the _states_ $y$ and $v_y$ smoothly vary in time (there's no teleportation or instantaneous changes in velocity).
Now we can fit a polynomial to the plot of each of these states with respect to time.
In this case, basic physics tells us that $v_y$ should be linear with time and $y$ should be quadratic with time.

Now for the simulation, lets solve a _boundary value problem_.
We know the initial and final height of the ball.
We know the initial velocity but not the terminal velocity.

_How long does it take the ball to reach the ground from an initial height of 100 m?_

We will explain how this problem is solved using the two implicit simulation techniques in Dymos:  high-order Gauss-Lobatto collocation, and the Radau Pseudospectral Method.

## Finding the trajectory using high-order Legendre-Gauss-Lobatto collocation

Using a 3rd-order polynomial segment to mimic the trajectory, we have two states and thus two polynomials.
Legendre-Gauss-Lobatto (LGL) collocation uses knowledge of the states and state-rates to form an interpolating polynomials.
A third-order polynomial requires four pieces of information to define it.
We'll take two points along the trajectory, and use those to guess the state values.
In Dymos, these points where state values are provided are called the _state discretization nodes_.
In LGL collocation, these points are chosen to be the LGL _nodes_ in the dimensionless time of the segment, which we call $\tau_s$.

Our initial guess for the trajectory might look something like this:

| Variables  | Description       | Value          | Notes                  |
|------------|-------------------|----------------|------------------------|
| $y_0$      | initial height    | 100 m          | fixed                  |
| $y_f$      | final height      | 0 m            | fixed                  |
| $v_{y0}$   | initial velocity  | 0 m/s          | fixed                  |
| $v_{yf}$   | final velocity    | -50 m/s        | free                   |
| $t_0$      | initial time      | 0              | fixed                  |
| $t_d$      | time duration     | 5              | free                   |
| $g$        | grav. acceleration| 9.80665 m/s^2  | assumed                |

To form 3rd-order polynomials for the states we need four pieces of information, but so far we only have the two endpoint values.
We'll use the state rates at the state discretization nodes to provide the other two values needed for interpolation.
To obtain these, we evaluate our ODE at the state discretization nodes.

\begin{align}
    \dot{y}_0 &= 0 \\
    \dot{y}_f &= -50 \\
    \dot{v}_{y0} &= -9.80665 \\
    \dot{v}_{yf} &= -9.80665
\end{align}

[IMAGE OF INITIAL TRAJECTORY]

But how do we know that our assumed trajectory is correct?
In LGL collocation, we check the interpolated slope of the polynomial at a 3rd point (the collocation node) and compare it to an evaluation of the ODE at the same point.
If the difference between the two is sufficiently close to zero, we can be somewhat confident that our polynomial is a reasonable match for the true time-history.
We call this difference in slope the collocation _defect_.

Having computed the ODE at the state discretization nodes, we can form interpolants for the values and the slope of the states at the collocation nodes.
Dymos uses Lagrange interpolants, such that the inputs to the system are the values of the states and controls, rather than less intuitive coefficients.
In this case, the state values at our single collocation node are:

\begin{align}
    y_m &= 50 \\
    v_{ym} &= -25
\end{align}

The polynomial slopes at the collocation node are:

\begin{align}
    y'_m &= 50 \\
    v'_{ym} &= -25
\end{align}

Now, having the state and time values at the collocation nodes, we can again evaluate the ODE.

\begin{align}
    \dot{y}_m &= -17 \\
    \dot{v}_{ym} &= -9.80665
\end{align}

In this case the known variables are the value of the polynomial at the left endpoint ($y_0$), and the value of the independent variable at the left and right endpoints of the interval to be fitted ($x_0$, $x_1$)
In this case, there are no _control_ inputs to the ODE.
However, if there were, the value of the control would also need to be provided at the collocation node.
Thus, in high-order Gauss-Lobatto collocation, the _control input nodes_ include both the state discretization nodes and the collocation nodes.

<!--Lets assume the following problem:  Fit a curve of some shape where only the value ($y$) of the curve is known at the left endpoint ($y_0$), but the slope of the curve ($\dot{y}$) may be computed anywhere.-->
<!--Find the final value along the curve $(y_f)$ at the end of the polynomials interval.-->

<!--Now we make a key assumption:  The shape can be accurately fit with a quadratic polynomial.-->
<!--In practice, we can expand this out to extremely high orders, but for now it keeps the example reasonably simple.-->
<!--A quadratic polynomial is defined by three points, hence our implicit method will involve guessing values of the polynomial at three points in the interval.-->
<!--For simplicity, let's define them at the endpoints and midpoint of the interval.-->
<!--THus we have the following variables and constraints:-->


<!--| Variables  | Description     | Constraints    | Description            |-->
<!--|------------|-----------------|----------------|------------------------|-->
<!--| $x_0$      | initial $x$     | $x_0 = 0$      | initial $x$ is known   |-->
<!--| $x_f$      | final $x$       | $x_f = 1$      | final $x$ is known     |-->
<!--| $y_0$      | initial value   | $y_0 = 3$      | initial value given    |-->
<!--| $y_m$      | midpoint value  | ?              |                        |-->
<!--| $y_f$      | final value     | ?              |                        |-->

<!--In this case, five independent variables are matched to only three equations of constraint.-->
<!--In order to have a well-posed problem with a single solution, two more constraints are needed.-->

<!--Recall, a function is available that returns the slope (derivative) of the polynomial:-->

<!--\begin{align}-->
<!--    \frac{dy}{dx} = f(x, y) \\-->
<!--\end{align}-->

<!--If $y_m$ and $y_f$ can be found such that the slope of the polynomial ($y\prime$) matches the computed derivative ($\frac{dy}{dx}$), then our polynomial should match the intended shape.-->

<!--\begin{align}-->
<!--    \Delta y_m &= y\prime_m - f(x_m, y_m) \\-->
<!--    \Delta y_f &= y\prime_f - f(x_f, y_f) \\-->
<!--\end{align}-->

<!--## Lagrange Polynomial Interpolation-->

<!--The lagrange interpolating polynomial across the interval for three given values is:-->

<!--\begin{align}-->
<!--  L(x) &= \sum_{i=0}^{k} y_i \prod_{j=0\\j \ne i}^{k} \frac{x-x_j}{x_i - x_j} \\-->
<!--  L(x) &= y_0 \frac{x - x_m}{x_0 - x_m} \frac{x - x_f}{x_0 - x_f} +-->
<!--          y_m \frac{x - x_0}{x_m - x_0} \frac{x - x_f}{x_m - x_f} +-->
<!--          y_f \frac{x - x_0}{x_f - x_0} \frac{x - x_m}{x_f - x_m}-->
<!--\end{align}-->

<!--But in this case, the values of $x$ at which we're interpolating the data are known:-->

<!--\begin{align}-->
<!--  L(x_0) &= y_0 \frac{x_0 - x_m}{x_0 - x_m} \frac{x_0 - x_f}{x_0 - x_f} +-->
<!--            y_m \frac{x_0 - x_0}{x_m - x_0} \frac{x_0 - x_f}{x_m - x_f} +-->
<!--            y_f \frac{x_0 - x_0}{x_f - x_0} \frac{x_0 - x_m}{x_f - x_m} \\-->
<!--  L(x_f) &= y_0 \frac{x_f - x_m}{x_0 - x_m} \frac{x_f - x_f}{x_0 - x_f} +-->
<!--            y_m \frac{x_f - x_0}{x_m - x_0} \frac{x_f - x_f}{x_m - x_f} +-->
<!--            y_f \frac{x_f - x_0}{x_f - x_0} \frac{x_f - x_m}{x_f - x_m} \\-->
<!--\end{align}-->

<!--The interpolated values at the endpoints are just a matrix-vector product.-->
<!--And the interpolation matrix can be computed and saved if the location of the interpolation points ($x_j$) are fixed.-->
<!--In this case, since the values are being interpolated at the same points at which values are given, the interpolation matrix is an identity matrix.-->

<!--\begin{align}-->
<!--   \begin{bmatrix} y_0 \\ y_f \end{bmatrix} &= \left[ L \right] \begin{bmatrix} y_0 \\ y_f \end{bmatrix} \\-->
<!--   \left[ L \right] &= \left[ I \right]-->
<!--\end{align}-->

<!--## Differentiating the interpolating polynomial-->

<!--If the interpolating polynomials are differentiated w.r.t. $x$, we can obtain a differentiation matrix that provides the _derivative_ of $y$ at the requested points:-->

<!--\begin{align}-->
<!--   \begin{bmatrix} y\prime_0 \\ y\prime_f \end{bmatrix} &= \left[ D \right] \begin{bmatrix} y_0 \\ y_f \end{bmatrix} \\-->
<!--\end{align}-->

<!--Now the final two constraints in the polynomial fitting can be expressed as the vector-valued function:-->

<!--\begin{align}-->
<!--    \Delta \bar{y} &= y\prime_m(\bar{y}) - f(\bar{x}, \bar{y}) \\-->
<!--\end{align}-->

<!--Typically, $\Delta \bar{y}$ are referred to as the _defect_ constraints.-->

<!--## Solution procedure-->

<!--The solution procedure for the curve fitting problem is an iterative process:-->

<!--1.  Guess values for $\bar{y}$.-->
<!--2.  Assess the constraints.-->
<!--3.  If the constraints are satisfied, the process is complete-->
<!--4.  If the constraints are not satisfied, find the derivative of the constraints w.r.t. $\bar{y}$ and use a gradient-based approach to propose a new value for $\bar{y}$.-->

<!--Steps 1-4 represent a gradient-based approach to implicitly fitting the data points.-->

<!--## What can go wrong?-->

<!--In the above example, we assumed that the shape could be fit to a quadratic equation.-->
<!--What do we do for shapes with higher orders?-->

<!--There are two options.-->
<!--We use a higher order polynomial (for instance, specifying $y$ at 5 points and constraing defects at 4 points).-->

<!--Alternatively, we can use multiple polynomial _segments_ across the interval.-->
<!--For instance, with three quadratic segments the differentiation matrix has the form. **TODO**-->

<!--The number and order of segments used to represent the fitting interval is known as the _grid_.-->
<!--It can be difficult to know what a sufficient grid is _a priori_.-->
<!--Fortunately, implicit collocation tools like Dymos typically provide automated grid refinement that will attempt to assess the fitting error and suggest a new grid until a suitable accuracy is achieved.-->

<!--## Exploiting sparsity-->

<!--One advantage of the multi-segment approach is that the interpolation and differentiation matrices are now _sparse_.-->
<!--That is, the values and derivatives in any segment are only dependent on the variables which affect that segment.-->

<!--\begin{align}-->
<!--\left[ D \right] &=-->
<!--\begin{bmatrix}-->
<!--  \left[ D_0 \right] & \left[ 0 \right] & \left[ 0 \right] \\-->
<!--  \left[ 0 \right] & \left[ D_1 \right] & \left[ 0 \right] \\-->
<!--  \left[ 0 \right] & \left[ 0 \right] & \left[ D_2 \right]-->
<!--\end{bmatrix}-->
<!--\end{align}-->

<!--Some nonlinear solvers and optimizers can capitalize upon this sparsity to solve the problem much more efficiently.-->
