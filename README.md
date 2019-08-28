# pyme-colored-image
PYME module for renderering 2D images with a color channel.

* More precise with color channel mapping.
	* Continuous color mapping c.f. max projection of 3D stack.
	* Can obey 'depth' opacity by setting alpha > 0.
	* Setting alpha to 0 equates to max projection.
	
* Slow.
	* Gaussian renderering currently does not support multiprocessing. Scales poorly with number of localizations. Can be improved.
	* Multiprocessing for Histogram renderering not optimized. Scales poorly with dense samples (counts/pixel). Potentially slower than Gaussian renderering with sigma < 1 pixel.

* Should be implemented on GPU. Marginal gains in image control (defined pixel size and capability for max projection) over regular point sprite display for significant cost in speed.
	