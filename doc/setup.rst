.. _setup:

Setting up
==========

To set up the package, just run the following lines in a console:

.. code-block:: shell

  git clone https://github.com/fastprof-hep/fastprof.git
  cd fastprof
  git checkout v0.2.0 -b local_branch  # use the current recommended version
  source ./setup-env.sh                # set up the environment
  
The last line sets up a python3 env, and installs required python packages.

Once the installation has been performed as described above, sessions in a new shell can be set up by running

.. code-block:: shell

  cd fastprof
  source ./setup-env.sh


  
