Installation
============

Install Using pip
-----------------

To install Dymos into site-packages of your local python installation, simply run

.. code-block:: console

    pip install git+https://github.com/OpenMDAO/dymos.git

Developer Mode
--------------

To install Dymos in developer mode, which will allow the package to be modified,
follow these two steps:

1. In terminal, clone the repository using `git`.

2. Go into the Dymos directory and use the command :code:`pip install -e .` to install dymos.

.. code-block:: console

    git clone https://github.com/OpenMDAO/dymos.git ./dymos.git
    cd dymos.git
    pip install -e .

Uninstalling
------------

If you want to uninstall Dymos, use the command :code:`pip uninstall dymos`.
