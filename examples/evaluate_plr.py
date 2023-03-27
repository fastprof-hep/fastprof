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
      "Requirement already satisfied: pandas>=1.1.0 in /home/nberger/anaconda3/lib/python3.8/site-packages (from fastprof) (1.1.3)\n",
      "Requirement already satisfied: mock>=4.0.3 in /home/nberger/anaconda3/lib/python3.8/site-packages (from fastprof) (5.0.1)\n",
      "Requirement already satisfied: scipy>=1.5.0 in /home/nberger/.local/lib/python3.8/site-packages (from fastprof) (1.8.1)\n",
      "Requirement already satisfied: sphinx-rtd-theme>=0.5.1 in /home/nberger/anaconda3/lib/python3.8/site-packages (from fastprof) (1.2.0)\n",
      "Requirement already satisfied: PyYAML>=5.1 in /home/nberger/anaconda3/lib/python3.8/site-packages (from fastprof) (5.3.1)\n",
      "Requirement already satisfied: sphinx-argparse>=0.2.5 in /home/nberger/anaconda3/lib/python3.8/site-packages (from fastprof) (0.4.0)\n",
      "Requirement already satisfied: numpy>=1.19.5 in /home/nberger/.local/lib/python3.8/site-packages (from fastprof) (1.22.1)\n",
      "Requirement already satisfied: matplotlib>=3.3.3 in /home/nberger/anaconda3/lib/python3.8/site-packages (from fastprof) (3.7.1)\n",
      "Requirement already satisfied: cycler>=0.10 in /home/nberger/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (0.10.0)\n",
      "Requirement already satisfied: kiwisolver>=1.0.1 in /home/nberger/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (1.3.0)\n",
      "Requirement already satisfied: packaging>=20.0 in /home/nberger/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (20.4)\n",
      "Requirement already satisfied: pillow>=6.2.0 in /home/nberger/.local/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (9.1.1)\n",
      "Requirement already satisfied: importlib-resources>=3.2.0 in /home/nberger/.local/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (5.8.0)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in /home/nberger/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (1.0.7)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in /home/nberger/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (4.39.2)\n",
      "Requirement already satisfied: python-dateutil>=2.7 in /home/nberger/.local/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (2.8.2)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in /home/nberger/anaconda3/lib/python3.8/site-packages (from matplotlib>=3.3.3->fastprof) (2.4.7)\n",
      "Requirement already satisfied: pytz>=2017.2 in /home/nberger/anaconda3/lib/python3.8/site-packages (from pandas>=1.1.0->fastprof) (2020.1)\n",
      "Requirement already satisfied: sphinx>=1.2.0 in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx-argparse>=0.2.5->fastprof) (3.2.1)\n",
      "Requirement already satisfied: docutils<0.19 in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx-rtd-theme>=0.5.1->fastprof) (0.16)\n",
      "Requirement already satisfied: sphinxcontrib-jquery!=3.0.0,>=2.0.0 in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx-rtd-theme>=0.5.1->fastprof) (4.1)\n",
      "Requirement already satisfied: six in /home/nberger/anaconda3/lib/python3.8/site-packages (from cycler>=0.10->matplotlib>=3.3.3->fastprof) (1.15.0)\n",
      "Requirement already satisfied: zipp>=3.1.0 in /home/nberger/.local/lib/python3.8/site-packages (from importlib-resources>=3.2.0->matplotlib>=3.3.3->fastprof) (3.8.1)\n",
      "Requirement already satisfied: sphinxcontrib-qthelp in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.0.3)\n",
      "Requirement already satisfied: snowballstemmer>=1.1 in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (2.0.0)\n",
      "Requirement already satisfied: sphinxcontrib-htmlhelp in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.0.3)\n",
      "Requirement already satisfied: sphinxcontrib-devhelp in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.0.2)\n",
      "Requirement already satisfied: babel>=1.3 in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (2.8.1)\n",
      "Requirement already satisfied: alabaster<0.8,>=0.7 in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (0.7.12)\n",
      "Requirement already satisfied: requests>=2.5.0 in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (2.24.0)\n",
      "Requirement already satisfied: sphinxcontrib-serializinghtml in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.1.4)\n",
      "Requirement already satisfied: Pygments>=2.0 in /home/nberger/.local/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (2.12.0)\n",
      "Requirement already satisfied: sphinxcontrib-applehelp in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.0.2)\n",
      "Requirement already satisfied: sphinxcontrib-jsmath in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.0.1)\n",
      "Requirement already satisfied: setuptools in /home/nberger/.local/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (63.2.0)\n",
      "Requirement already satisfied: Jinja2>=2.3 in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (2.11.2)\n",
      "Requirement already satisfied: imagesize in /home/nberger/anaconda3/lib/python3.8/site-packages (from sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.2.0)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in /home/nberger/anaconda3/lib/python3.8/site-packages (from Jinja2>=2.3->sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.1.1)\n",
      "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /home/nberger/anaconda3/lib/python3.8/site-packages (from requests>=2.5.0->sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (1.25.11)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /home/nberger/anaconda3/lib/python3.8/site-packages (from requests>=2.5.0->sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (2020.6.20)\n",
      "Requirement already satisfied: chardet<4,>=3.0.2 in /home/nberger/anaconda3/lib/python3.8/site-packages (from requests>=2.5.0->sphinx>=1.2.0->sphinx-argparse>=0.2.5->fastprof) (3.0.4)\n",
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
      "{'xs_BSM': <fastprof.base.ModelPOI object at 0x7f9c19368850>}\n"
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
      "{'np_lumi': <fastprof.base.ModelNP object at 0x7f9c19385040>, 'np_eff': <fastprof.base.ModelNP object at 0x7f9c193850d0>, 'nBkg': <fastprof.base.ModelNP object at 0x7f9c19385160>}\n"
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
    "pars = Parameters([poi_value], model=model)"
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
    "print(pars)"
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
    "print('NLL(%s = %g, nominal NPs) = %g' % (model.poi(0).name, poi_value, model.nll(pars, data)))"
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
    "print(NPMinimizer(data).profile(pars))"
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
     "ename": "NameError",
     "evalue": "name 'par' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Input \u001b[0;32mIn [12]\u001b[0m, in \u001b[0;36m<cell line: 1>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mNLL(\u001b[39m\u001b[38;5;132;01m%s\u001b[39;00m\u001b[38;5;124m = \u001b[39m\u001b[38;5;132;01m%g\u001b[39;00m\u001b[38;5;124m, nominal NPs) = \u001b[39m\u001b[38;5;132;01m%g\u001b[39;00m\u001b[38;5;124m'\u001b[39m \u001b[38;5;241m%\u001b[39m (model\u001b[38;5;241m.\u001b[39mpoi(\u001b[38;5;241m0\u001b[39m)\u001b[38;5;241m.\u001b[39mname, poi_value, NPMinimizer(data)\u001b[38;5;241m.\u001b[39mprofile_nll(\u001b[43mpar\u001b[49m)))\n",
      "\u001b[0;31mNameError\u001b[0m: name 'par' is not defined"
     ]
    }
   ],
   "source": [
    "print('NLL(%s = %g, nominal NPs) = %g' % (model.poi(0).name, poi_value, NPMinimizer(data).profile_nll(pars)))"
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
   "execution_count": null,
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
   "execution_count": null,
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
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.plot(poi_vals, profile_nll_vals)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now compute the gradients and hessians. First compute the profile value of the parameters:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "prof_pars = NPMinimizer(data).profile_nll(pars)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now the gradient (of course just a size-1 vector here) :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(model.gradient(pars, data))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "And the Hessian (just 1x1 matrix here) :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(model.hessian(pars, data))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
