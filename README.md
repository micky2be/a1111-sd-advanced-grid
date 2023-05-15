# Stable Diffusion Advanced Grid

## Status
**Under development:** _limited features and some bugs may arise._

## Installation

 1. Install extension by going to Extensions tab -> Install from URL -> Paste github URL and click Install.
 2. After it's installed, go back to the Installed tab in Extensions and press Apply and restart UI.
 3. Installation finished.

## Concept and Idea

Extension for the [AUTOMATIC1111 Stable Diffusion WebUI][] to generate multidimensional grids.
The current grid system can be limited and can be heavy (as for now the grid image is always generated in memory)

## Limitation

As this is multidimensional there are no direct ways to generate a grid image like the "X/Y/Z plot" script does.
Keep in mind having multidimension as an exponential factor of varation for your rendering.
As for now, the script works only under `text2image` until further development and testing would be done.

## How to use
Most fields should work as it would in the [X/Y/Z plot][].
The output will go into a subfolder of the defined grid folder.

You can resume a generation if necessary, or add varation to your grid. The script will detect existing image generated previously and will skip them.
This will work only if you add variation to existing axes. A new axis will trigger a new version.
Currently, a change outside of the axes will not be recognise. If you need to make a new grid, make sure to change the name of it.

## Expansion and hooks
**TBD**


[AUTOMATIC1111 Stable Diffusion WebUI]: https://github.com/AUTOMATIC1111/stable-diffusion-webui
[X/Y/Z plot]: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#xyz-plot