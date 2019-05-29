# xBIT
an easy to use scanning tool with machine learning abilities

## Purpose
``xBIT`` is a tool for performing parameter scans in beyond the Standard Model (BSM) theories. It's written in ``python`` and fully open source. The main focus was to provide an easy to use tool to perform the daily tasks of a BSM phenomenologist: exploring the parameter space of new models. It was developed under the impression of the SARAH/SPheno framework, but should be useable with other tools that use the SLHA format to transfer 
data. Out-of-the-box it also supports ``MicrOmegas`` for dark matter scans, ``HiggsBounds`` and ``HiggsSignals`` for checking the Higgs properties and Vevacious for testing the vacuum stability. Classes for other tools can be defined which are then automatically included in the scans. Also new scan classes can be added in a modular way.

In order to improve the efficiency of the parameter scans,  the recently proposed 'Machine Learning  Scan' (MLS) approach is included. For this purpose, ``xBIT`` trains artificial neural networks which are generated by linking ``pyTorch``.

## Running
A scan is started via
```
python3 xBIT INPUTFILE
```
where the INPUTFILE contains all information to define a scan. An example for a CMSSM scenario looks like
```
{
  "Setup": {
    "Settings": "MSSM.json",
    "Name": "m0m12_grid",
    "Type": "Grid",
    "Cores": 1
  },

  "Included_Codes": {
    "HiggsBounds": "True",
    "HiggsSignals": "True",
    "MicrOmegas": "True",
    "Vevacious": "False"
  },

  "Variables": {
      "0": "LOG(100,10000,10)",
      "1": "LINEAR(200,500,10)",
      "2": "LINEAR(10,20,2)"
  },

  "Blocks": {
    "MODSEL": {
         "1": 1,
         "2": 1,
         "3": 1
    },

    "SMINPUTS": {
         "2": 1.166370E-05,
         "3": 1.187000E-01,
         "4": 9.118870E+01,
         "5": 4.180000E+00,
         "6": 1.735000E+02,
         "7": 1.776690E+00
    },

    "MINPAR": {
         "1": "2.0*Variable[0]",
         "2": "Variable[0]",
         "3": "Variable[2]",
         "4": 1.0,
         "5": "Variable[1]"
    },

  "SPhenoInput": {
...
  }
}
}
```

## Note
This package is completely independent of another scanning tool which comes with a bunch of 'BITs'. Nevertheless, the name was chosen on purpose to point out that it follows a philosophy which is orthogonal to this other tool.
