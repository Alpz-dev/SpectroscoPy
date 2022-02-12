# SpectroscoPy

This project is a set of tools that aims to make the importing, processing, analyzing, and plotting of various spectroscopic data easy and simple to use. 

It is primarily built around data objects, with each imported dataset being stored as an object containing an multitude of data processing functions as methods in its class. 

All data processing methods are **NON-DESTRUCTIVE** and **DO NOT ALTER THE ORIGINAL DATA FILE IN ANY WAY**.

Manipulated data can be exported to both .csv and tab-delimted .txt files with optional header information.


Currently implemented data processing functions:
  - translate, scale, transpose, crop
  - +, -, *, /, ln, log
  - differentiate/integrate
  - peak/shoulder detection
  - gaussian/lorentzian deconvolution with parameter memoization
  - exponential fitting
  - linear regression
