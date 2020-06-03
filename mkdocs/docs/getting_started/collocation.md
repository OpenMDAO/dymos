# What is collocation?

Dymos is based on an implicit integration technique, collocation.

The _state_ of a dynamic system defines its configuration at a point in time.
Common state variables for dynamic systems are position, velocity, and mass.
Given an initial state, a time duration, and ordinary differential equations that define the evolution of the states in time, the state of the system at the end of the time duration can be determined.
This is known as the _initial value problem_ (IVP) and there are numerous techniques and software packages that can be used to solve the IVP.
The initial value problem is typically the basis for the simulation of systems.
This time-marching approach is the basis for _shooting methods_ in optimization.
While Dymos supports shooting methods, its focus is on implicit collocation techniques, for performance reasons.

First, a few assumptions about the dynamic system:

1. Over portions of the trajectory, the state is continuous (there are no instantaneous changes in the states)
2. Over portions of the trajectory, the state is continuous in its first derivative w.r.t. time. (there are no instantaneous changes in the state rates)

If those two conditions are met, then it's reasonable fit a polynomial to a plot of the state value over time.
Given the initial or final value of a state, and values of its rate at some number of points across the trajectory, then a polynomial can be fit whose initial or final value matches, and whose rates match the sampled rates.
The value of the state at any point throughout the trajectory can then be inferred by interpolating the collocated polynomial.

Solving this trajectory implicitly basically means _guessing_ values for the states along the trajectory.
While one could use a power series polynomial coefficients ($a t^2 + b t + c), these coefficients can have vastly different values depending on the shape of the curve, making them difficult to scale numerically.
Instead, Lagrange interpolating polynoials are typically used, where the actual values to be interpolated at various points are the implicit variables.
This makes scaling the problem easier, and gives the implicit variables more intuitive meaning, making it easier to provide an initial guess.

## Example:  Fitting a polynomial over an interval

As an example, lets assume we are trying to fit a polynomial to some unknown shape over an iterval.
In this case the known variables are the value of the polynomial at the left endpoint ($y_0$), and the value of the independent variable at the left and right endpoints of the interval to be fitted ($x_0$, $x_1$)

Lets assume the following problem:  Fit a curve of some shape where only the value ($y$) of the curve is known at the left endpoint ($y_0$), but the slope of the curve ($\dot{y}$) may be computed anywhere.  
Find the final value along the curve $(y_f)$ at the end of the polynomials interval.

Now we make a key assumption:  The shape can be accurately fit with a quadratic polynomial.
In practice, we can expand this out to extremely high orders, but for now it keeps the example reasonably simple.
A quadratic polynomial is defined by three points, hence our implicit method will involve guessing values of the polynomial at three points in the interval.
For simplicity, let's define them at the endpoints and midpoint of the interval.
THus we have the following variables and constraints:


| Variables  | Description     | Constraints    | Description            |
|------------|-----------------|----------------|------------------------|
| $x_0$      | initial $x$     | $x_0 = 0$      | initial $x$ is known   |
| $x_f$      | final $x$       | $x_f = 1$      | final $x$ is known     |
| $y_0$      | initial value   | $y_0 = 3$      | initial value given    |
| $y_m$      | midpoint value  | ?              |                        |
| $y_f$      | final value     | ?              |                        |

In this case, five independent variables are matched to only three equations of constraint.
In order to have a well-posed problem with a single solution, two more constraints are needed.

Recall, a function is available that returns the slope (derivative) of the polynomial:

\begin{align}
    \frac{dy}{dx} = f(x, y) \\
\end{align}

If $y_m$ and $y_f$ can be found such that the slope of the polynomial ($y\prime$) matches the computed derivative ($\frac{dy}{dx}$), then our polynomial should match the intended shape.

\begin{align}
    \Delta y_m &= y\prime_m - f(x_m, y_m) \\
    \Delta y_f &= y\prime_f - f(x_f, y_f) \\
\end{align}

## Lagrange Polynomial Interpolation

The lagrange interpolating polynomial across the interval for three given values is:

\begin{align}
  L(x) &= \sum_{i=0}^{k} y_i \prod_{j=0\\j \ne i}^{k} \frac{x-x_j}{x_i - x_j} \\
  L(x) &= y_0 \frac{x - x_m}{x_0 - x_m} \frac{x - x_f}{x_0 - x_f} +
          y_m \frac{x - x_0}{x_m - x_0} \frac{x - x_f}{x_m - x_f} +
          y_f \frac{x - x_0}{x_f - x_0} \frac{x - x_m}{x_f - x_m}
\end{align}

But in this case, the values of $x$ at which we're interpolating the data are known:

\begin{align}
  L(x_0) &= y_0 \frac{x_0 - x_m}{x_0 - x_m} \frac{x_0 - x_f}{x_0 - x_f} +
            y_m \frac{x_0 - x_0}{x_m - x_0} \frac{x_0 - x_f}{x_m - x_f} +
            y_f \frac{x_0 - x_0}{x_f - x_0} \frac{x_0 - x_m}{x_f - x_m} \\
  L(x_f) &= y_0 \frac{x_f - x_m}{x_0 - x_m} \frac{x_f - x_f}{x_0 - x_f} +
            y_m \frac{x_f - x_0}{x_m - x_0} \frac{x_f - x_f}{x_m - x_f} +
            y_f \frac{x_f - x_0}{x_f - x_0} \frac{x_f - x_m}{x_f - x_m} \\
\end{align}

The interpolated values at the endpoints are just a matrix-vector product.
And the interpolation matrix can be computed and saved if the location of the interpolation points ($x_j$) are fixed.
In this case, since the values are being interpolated at the same points at which values are given, the interpolation matrix is an identity matrix.

\begin{align}
   \begin{bmatrix} y_0 \\ y_f \end{bmatrix} &= \left[ L \right] \begin{bmatrix} y_0 \\ y_f \end{bmatrix} \\
   \left[ L \right] &= \left[ I \right]
\end{align}

## Differentiating the interpolating polynomial

If the interpolating polynomials are differentiated w.r.t. $x$, we can obtain a differentiation matrix that provides the _derivative_ of $y$ at the requested points:

\begin{align}
   \begin{bmatrix} y\prime_0 \\ y\prime_f \end{bmatrix} &= \left[ D \right] \begin{bmatrix} y_0 \\ y_f \end{bmatrix} \\
\end{align}

Now the final two constraints in the polynomial fitting can be expressed as the vector-valued function:

\begin{align}
    \Delta \bar{y} &= y\prime_m(\bar{y}) - f(\bar{x}, \bar{y}) \\
\end{align}

Typically, $\Delta \bar{y}$ are referred to as the _defect_ constraints.

## Solution procedure

The solution procedure for the iterative curve fitting problem is an iterative process:

1. Guess values for $\bar{y}$.
2. Assess the constraints.
3. If the constraints are satisfied, the process is complete
4. If the constraints are not satisfied, use a find the derivative of the constraints w.r.t. $\bar{y}$ and use a gradient-based approach to propose a new value for $\bar{y}$.

Steps 1-4 represent a gradient-based approach to implicitly fitting the data points.

## What can go wrong?

In the above example, we assumed that the shape could be fit to a quadratic equation.
What do we do for shapes with higher orders?

There are two options.
We use a higher order polynomial (for instance, specifying $y$ at 5 points and constraing defects at 4 points).

Alternatively, we can use multiple polynomial _segments_ across the interval.  
For instance, with three quadratic segments the differentiation matrix has the form.

The number and order of segments used to represent the fitting interval is known as the _grid_.
It can be difficult to know what a sufficient grid is _a priori_.
Fortunately, implicit collocation tools like Dymos typically provide automated grid refinement that will attempt to assess the fitting error and suggest a new grid until a suitable accuracy is achieved.

## Exploiting sparsity

One advantage of the multi-segment approach is that the interpolation and differentiation matrices are now _sparse_.
That is, the values and derivatives in any segment are only dependent on the variables which affect that segment.

\begin{align}
\left[ D \right] &=
\begin{bmatrix}
  \left[ D_0 \right] & \left[ 0 \right] & \left[ 0 \right] \\
  \left[ 0 \right] & \left[ D_1 \right] & \left[ 0 \right] \\
  \left[ 0 \right] & \left[ 0 \right] & \left[ D_2 \right]
\end{bmatrix}
\end{align}

Some nonlinear solvers and optimizers can capitalize upon this sparsity to solve the problem much more efficiently.
