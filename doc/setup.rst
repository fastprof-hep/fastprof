.. _setup:

Setting up
==========

From github
-----------

To set up the package, just run the following lines in a console:

.. code-block:: shell

  git clone https://github.com/fastprof-hep/fastprof.git
  cd fastprof
  git checkout v0.4.1 -b v0.4.1  # use the current recommended version
  source ./setup.sh              # set up the environment
  
The last line sets up a python3 env, and installs required python packages.

Once the installation has been performed as described above, sessions in a new shell can be set up by running

.. code-block:: shell

  cd fastprof
  source ./setup.sh

Using pip
---------

Alternatively, the package can be installed using `pip` :

.. code-block:: shell

   pip install fastprof
   
Currently this will install `v0.4.1`, the latest stable release.
