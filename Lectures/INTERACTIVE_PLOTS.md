# How the interactive plots work

Every interactive plot in these lectures is plain HTML and JavaScript rendering data that
the notebook itself computed. None of them uses `ipywidgets`, and none needs a kernel once
its cell has run.

That is not a stylistic choice. `ipywidgets` do not render at all in RISE fullscreen
([jupyterlab-rise#119](https://github.com/jupyterlab-contrib/rise/issues/119)), and
JupyterLab mis-routes widget output during *Restart Kernel and Run All*, which used to put
a figure from one cell underneath a different one. Plain HTML has neither problem, and also
survives Colab, exported HTML, a dead kernel and an untrusted notebook.

## The shape of a plot

Each one is three or four cells:

| cell | `slide_type` | what it is |
|---|---|---|
| definition | `notes` | your plotting function, unchanged |
| static | `skip` | calls that function once and stores a PNG |
| builder | `notes` | computes the arrays and builds the HTML |
| display | `slide` | one line, e.g. `HTML(_joint_pdf_html)` |

**The static cell is not decoration.** It is what GitHub, nbviewer and a PDF export show,
because those strip JavaScript and the interactive version then draws nothing. It is also
the cross-check: the two plots are drawn from the same numbers, so if they ever disagree,
something in the handover is wrong.

## What to change, and where

**In Python — the builder cell.** Legend text, slider ranges and steps, panel titles,
figure width and height, display resolution. These are ordinary Python dicts; edit them
there.

**In Python — your own code.** Anything about *what the plot shows*. Change `sigma_z`, or
`x`, or the model itself, and both the static and the interactive plot follow, because the
renderer draws whatever array it is handed and does not know the model.

**In JavaScript — the renderer.** Only appearance: adding an annotation, changing a marker
shape, altering the axis styling. This is the one place that costs real effort, and it is
deliberately the smallest part.

## How the data gets across

Three patterns, depending on what the slider does.

**The slider slices a fixed field.** Inject the field. Lecture 4's joint pdf hands over
`joint_pdf_y_z_grid` as 16-bit integers; the contour rings are marching squares on that
array and the conditional slices are a row and a column of it. No formula in the renderer.

**The slider moves a marker over a fixed curve.** Inject the curve. The Dirac-delta plots
in Lecture 8 hand over the posterior, the mode and mean positions, and the slider bounds.
No formula in the renderer.

A variant: in Lecture 22's regression PPD the density has a *fixed shape* because its
standard deviation does not depend on `x`, so one profile is injected and the renderer
slides it along the mean curve. That is exact here and would be wrong the moment sigma
became a function of `x`.

**The slider changes the model.** Then there is no fixed field, and a formula has to live
in the renderer. This happens twice in the whole course: a sigmoid in Lecture 22's
Bernoulli plot, and closed-form least squares in Lecture 11's outlier demo. Both are small
enough to read at a glance, and the least-squares one was checked against
`np.linalg.lstsq` to ten decimal places.

## Slider steps

An HTML range input yields `min + k·step`, so a step of 0.01 will not land on a mean that
is not a multiple of 0.01. Where it matters, the grid is anchored on the mean:

```js
var lo = P.mean_z - Math.ceil((P.mean_z - x0)/st)*st;
```

which makes the mean exactly reachable, and the mode too when it sits a whole number of
steps away. That is what lets the Dirac delta land precisely on both.

## Two things that will bite

**`plt.close(fig)` in a callback is right for the inline backend and wrong under ipympl.**
With `%matplotlib inline` the figure is a picture and closing it prevents a leak. Under
`%matplotlib widget` the figure *is* the widget, and closing it destroys the canvas.

**A notebook must be trusted for stored JavaScript to run.** The frames and canvases are
assigned by script, so an untrusted notebook shows the slider chrome and no figure. Running
the notebook yourself trusts it; otherwise `jupyter trust <notebook>`. The signature is
per-machine and does not travel in the file — which is why the static cells exist.

## Do not use Run All on these lectures

Not for the plots any more — they are immune. But the notebooks still contain ordinary
matplotlib figures, and the same output-routing race can misplace those. Open the notebook
and run down it in order.

---

*Written by Claude (Opus 5) with Miguel Bessa, September 2026, alongside the renderers it
describes. Recorded here because the approach did not come from either side alone: the
failure took several wrong turns to diagnose, and computing the plots in the browser only
surfaced after Miguel refused to accept that a three-slider plot could not be interactive.
If something here is wrong, the static plot beside each interactive one is the thing to
trust.*
