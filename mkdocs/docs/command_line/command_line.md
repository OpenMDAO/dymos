# Dymos Command Line Interface

Dymos has several command line argumants that can make it easier to repeatedly run
a script with different options.

You can see all the possible Dymos command line options by running `dymos --help`:

```
dymos --help
usage: dymos [-h] [-s] [-n] [-i] [-t SOLUTION] [-r] [-l REFINE_LIMIT] script

Dymos Command Line Tool

positional arguments:
  script                Python script that creates a Dymos problem to run

optional arguments:
  -h, --help            show this help message and exit
  -s, --simulate        If given, perform simulation after solving the
                        problem.
  -n, --no_solve        If given, do not run the driver on the problem
                        (simulate only)
  -i, --no_iterate      If given, do not iterate the driver on the problem)
  -t SOLUTION, --solution SOLUTION
                        A previous solution record file (or explicit
                        simulation record file) from which the initial guess
                        should be loaded. (default: None)
  -r, --reset_grid      If given, reset the grid to the specs given in the
                        problem definition instead of the grid associated with
                        the loaded solution.
  -l REFINE_LIMIT, --refine_limit REFINE_LIMIT
                        The number of passes through the grid refinement
                        algorithm to use. (default: 0)
```

The only non-optional argument to a `dymos` command line invocation is the name of a script that
creates an instance of a Dymos Problem. For example:

=== "brachistochrone_for_command_line.py"
{{ inline_source('dymos.utils.test.brachistochrone_for_command_line',
include_def=True,
include_docstring=True,
indent_level=0)
}}

The Dymos command line handler recognizes the Dymos Problem by the call to _final_setup_,
so the script that creates a problem for command line execution should call that function last.

## Solving for the optimal trajectory

The default behavior for calling a script with the dymos command line is to solve the optimal control problem,
equivalent to calling the _problem.run_driver_ function. For example:

```dymos dymos/utils/test/brachistochrone_for_command_line.py```

Dymos will run the optimizer to solve the problem created by the script and show the results.

## Loading an existing trajectory as an initial guess

You will see a message before the run about a recorder being added. This is a database file called
`dymos_solution.db` (automaticaly created in your current working directory) that will allow restarting
work on the Dymos problem from where it left off. The `-t` or `--solution` command line option is used
to tell `dymos` to restart from the specified recorded solution:

```dymos -t dymos_solution.db dymos/utils/test/brachistochrone_for_command_line.py```

If you run the two commands above, the second command will report that the solution was found in one iteration
(because it started from a converged solution that was already found by the first command).
This option is useful for combining with other command line options to continue simulating or refining a
solution from a previous command line involcation.

The name of the automatially created recorder database is always `dymos_solution.db`. If you restart
using a database with that name (as above), the database being read will be renamed to `old_dymos_solution.db`
before the run to avoid overwriting the input.

## Simulating a trajectory

There are two command line options related to simulating a trajectory (propaging the ODEs). The options are
equivalent to calling _problem.run_model_ in addition to or instead of _problem.run_driver_.

 The `-s` or `--simulate` command line option runs the simulation after the optimal control problem is solved.
 For example:

 ```dymos -s dymos/utils/test/brachistochrone_for_command_line.py```

 The `-n` or `--no_solve` command line option runs the simulation but skips running the driver to solve the optimal
 control problem. For example:

 ```dymos -n dymos/utils/test/brachistochrone_for_command_line.py```

## Using grid refinement

There are two command line options related to grid refinement.

The `-l` or `--refine_limit` command line option is used to set the number of passes through the grid refinement
algorithm. It defaults to zero, which does no grid refinement. Setting it to a positive integer will enable grid
refinement. For example:

```dymos -l 10 dymos/utils/test/brachistochrone_for_command_line.py```

The `-r` or `--reset_grid` command line option (_not currently implemented_) resets the grid to the specifications
given in the problem definition, ignoring any grid that would have been loaded from a restart database. It is called
like this:

```dymos -r -t dymos_solution.db dymos/utils/test/brachistochrone_for_command_line.py```
