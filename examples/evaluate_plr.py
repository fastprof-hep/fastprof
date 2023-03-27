{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*Simple jupyter notebook to demonstrate profile likelihood estimation*\n",
    "\n",
    "First, install the `fastprof` package locally"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: fastprof in /home/nberger/anaconda3/lib/python3.8/site-packages (0.4.1)\n",
      "Requirement already satisfied: PyYAML>=5.1 in /home/nberger/anaconda3/lib/python3.8/site-packages (from fastprof) (5.3.1)\n",
      "Requirement already satisfied: mock>=4.0.3 in /home/nberger/anaconda3/lib/python3.8/site-packages (from fastprof) (5.0.1)\n",
      "Requirement already satisfied: pandas>=1.1.0 in /home/nberger/anaconda3/lib/python3.8/site-packages (from fastprof) (1.1.3)\n",
      "Requirement already satisfied: sphinx-argparse>=0.2.5 in /home/nberger/anaconda3/lib/python3.8/site-packages (from fastprof) (0.4.0)\n",
      "Requirement already satisfied: numpy>=1.19.5 in /home/nberger/.local/lib/python3.8/site-packages (from fastprof) (1.22.1)\n",
      "Requirement already satisfied: matplotlib>=3.3.3 in /home/nberger/anaconda3/lib/python3.8/site-packages (from fastprof) (3.7.1)\n",
      "Requirement already satisfied: sphinx-rtd-theme>=0.5.1 in /home/nberger/anaconda3/lib/python3.8/site-packages (from fastprof) (1.2.0)\n",
      "Requirement already satisfied: scipy>=1.5.0 in /home/nberger/.local/lib/python3.8/site-packages (from fastprof) (1.8.1)\n",
      "Requirement already satisfied: importlib-resources>=3.2.0 in /home/nberger/.local/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (5.8.0)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in /home/nberger/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (4.39.2)\n",
      "Requirement already satisfied: python-dateutil>=2.7 in /home/nberger/.local/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (2.8.2)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in /home/nberger/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (1.0.7)\n",
      "Requirement already satisfied: kiwisolver>=1.0.1 in /home/nberger/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (1.3.0)\n",
      "Requirement already satisfied: pillow>=6.2.0 in /home/nberger/.local/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (9.1.1)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in /home/nberger/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (2.4.7)\n",
      "Requirement already satisfied: cycler>=0.10 in /home/nberger/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (0.10.0)\n",
      "Requirement already satisfied: packaging>=20.0 in /home/nberger/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (20.4)\n",
      "Requirement already satisfied: pytz>=2017.2 in /home/nberger/anaconda3/lib/python3.8/site-packages (from pandas>=1.1.0->fastprof) (2020.1)\n",
      "Requirement already satisfied: sphinx>=1.2.0 in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx-argparse>=0.2.5->fastprof) (3.2.1)\n",
      "Requirement already satisfied: sphinxcontrib-jquery!=3.0.0,>=2.0.0 in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx-rtd-theme>=0.5.1->fastprof) (4.1)\n",
      "Requirement already satisfied: docutils<0.19 in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx-rtd-theme>=0.5.1->fastprof) (0.16)\n",
      "Requirement already satisfied: six in /home/nberger/anaconda3/lib/python3.8/site-packages (from cycler>=0.10->matplotlib>=3.3.3->fastprof) (1.15.0)\n",
      "Requirement already satisfied: zipp>=3.1.0 in /home/nberger/.local/lib/python3.8/site-packages (from importlib-resources>=3.2.0->matplotlib>=3.3.3->fastprof) (3.8.1)\n",
      "Requirement already satisfied: sphinxcontrib-htmlhelp in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.0.3)\n",
      "Requirement already satisfied: Pygments>=2.0 in /home/nberger/.local/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (2.12.0)\n",
      "Requirement already satisfied: sphinxcontrib-devhelp in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.0.2)\n",
      "Requirement already satisfied: Jinja2>=2.3 in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (2.11.2)\n",
      "Requirement already satisfied: sphinxcontrib-serializinghtml in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.1.4)\n",
      "Requirement already satisfied: snowballstemmer>=1.1 in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (2.0.0)\n",
      "Requirement already satisfied: sphinxcontrib-applehelp in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.0.2)\n",
      "Requirement already satisfied: imagesize in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.2.0)\n",
      "Requirement already satisfied: requests>=2.5.0 in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (2.24.0)\n",
      "Requirement already satisfied: sphinxcontrib-qthelp in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.0.3)\n",
      "Requirement already satisfied: alabaster<0.8,>=0.7 in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (0.7.12)\n",
      "Requirement already satisfied: sphinxcontrib-jsmath in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.0.1)\n",
      "Requirement already satisfied: babel>=1.3 in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (2.8.1)\n",
      "Requirement already satisfied: setuptools in /home/nberger/.local/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (63.2.0)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in /home/nberger/anaconda3/lib/python3.8/site-packages (from Jinja2>=2.3->sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.1.1)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /home/nberger/anaconda3/lib/python3.8/site-packages (from requests>=2.5.0->sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (2020.6.20)\n",
      "Requirement already satisfied: chardet<4,>=3.0.2 in /home/nberger/anaconda3/lib/python3.8/site-packages (from requests>=2.5.0->sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (3.0.4)\n",
      "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /home/nberger/anaconda3/lib/python3.8/site-packages (from requests>=2.5.0->sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.25.11)\n",
      "Requirement already satisfied: idna<3,>=2.5 in /home/nberger/anaconda3/lib/python3.8/site-packages (from requests>=2.5.0->sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (2.10)\n",
      "\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m A new release of pip available: \u001b[0m\u001b[31;49m22.1.2\u001b[0m\u001b[39;49m -> \u001b[0m\u001b[32;49m23.0.1\u001b[0m\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m To update, run: \u001b[0m\u001b[32;49mpip install --upgrade pip\u001b[0m\n"
     ]
    }
   ],
   "source": [
    "import sys\n",
    "!pip install --prefix {sys.prefix} fastprof"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This should have generated a lot of output, but if all goes well the package should be installed.\n",
    "We can now import the classes we need\n",
    "- `Model` : the class holding the model PDF, used to evaluate negative log-likelihoods (NLL)\n",
    "- `Data` : the class holding the dataset\n",
    "- `Parameters` : class holding the values of the model parameters of interest (POIs) and nuisance parameters (NPs)\n",
    "- `NPMinimizer` : class performing minmization over the NPs, to obtain the _profile likelihood_ for a given value of the POIs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastprof import Model, Data, Parameters, NPMinimizer"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now instantiate the model and data from a model stored in a JSON file in this directory:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Model.create('multi_channel.json')\n",
    "data = Data(model).load('multi_channel.json')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "First inspect the model: \n",
    "- `model.pois` is where the definition of all POIs are stored (as a name -> object dict)\n",
    "- `model.nps` is the same for the NPs.\n",
    "\n",
    "Here we have only 1 POIs:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'xs_BSM': <fastprof.base.ModelPOI object at 0x7f42505d0580>}\n"
     ]
    }
   ],
   "source": [
    "print(model.pois)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Access the `ModelPOI` object and print it out:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "POI xs_BSM = 0 fb (min = 0 fb, max = 10 fb)\n"
     ]
    }
   ],
   "source": [
    "print(model.poi(0))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Same for the NPs, except we have 3 of them:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'np_lumi': <fastprof.base.ModelNP object at 0x7f42505d0fa0>, 'np_eff': <fastprof.base.ModelNP object at 0x7f42505cafd0>, 'nBkg': <fastprof.base.ModelNP object at 0x7f41d3a3fdc0>}\n"
     ]
    }
   ],
   "source": [
    "print(model.nps)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "NP np_lumi = 0 +/- 1 constrained to aux_lumi with σ = 1\n",
      "NP np_eff = 0 +/- 1 constrained to aux_eff with σ = 1\n",
      "NP nBkg = 64 +/- 8 free parameter\n"
     ]
    }
   ],
   "source": [
    "for npar in model.nps.values() : print(npar)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now evaluate the NLL for a given POI value, say 2. We create a `Parameters` object for this POI value by passing a list of length 1 (in the case of more POIs, the order in the list is the same as in `model.pois`). Since we don't specify NPs, they will be set to their nominal values (here 0)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "poi_value = 2\n",
    "par = Parameters([poi_value], model=model)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Print the `Parameters` object:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "POIs : xs_BSM       =   2.0000\n",
      "NPs  : np_lumi      =   0.0000 (unscaled :       0.0000)\n",
      "       np_eff       =   0.0000 (unscaled :       0.0000)\n",
      "       nBkg         =   0.0000 (unscaled :      64.0000)\n"
     ]
    }
   ],
   "source": [
    "print(par)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now evaluate the NLL:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "NLL(xs_BSM = 2, nominal NPs) = 4.69939\n"
     ]
    }
   ],
   "source": [
    "print('NLL(%s = %g, nominal NPs) = %g' % (model.poi(0).name, poi_value, model.nll(par, data)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This is the NLL for the nominal NP values. If we want to minimize with respect to the NPs (_profile_ the likelihood), we need to use the minimizer object:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "POIs : xs_BSM       =   2.0000\n",
      "NPs  : np_lumi      =  -0.1116 (unscaled :      -0.1116)\n",
      "       np_eff       =  -0.6458 (unscaled :      -0.6458)\n",
      "       nBkg         =  -0.4659 (unscaled :      60.2729)\n"
     ]
    }
   ],
   "source": [
    "print(NPMinimizer(data).profile(par))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The command above returns the parameter values at the minimum, to get the NLL at the minimum one uses `profile_nll` instead:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "NLL(xs_BSM = 2, nominal NPs) = 4.36641\n"
     ]
    }
   ],
   "source": [
    "print('NLL(%s = %g, nominal NPs) = %g' % (model.poi(0).name, poi_value, NPMinimizer(data).profile_nll(par)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As an illustration, perform a simple profile likelihood scan over $[-0.4, 2]$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy\n",
    "poi_vals = numpy.linspace(-0.4, 2, 21)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Compute the profile likelihood at each point:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "profile_nll_vals = [ NPMinimizer(data).profile_nll(Parameters([poi_val], model=model)) for poi_val in poi_vals ]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Plot the result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x7f41d39e0160>]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.plot(poi_vals, profile_nll_vals)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
