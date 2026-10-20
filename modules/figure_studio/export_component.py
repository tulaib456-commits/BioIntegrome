"""
Interactive Plotly chart with BOTH export options, done entirely in the
user's browser (no server-side Chrome / kaleido needed):

  * the camera icon in the chart's toolbar  -> high-resolution PNG
  * a floppy-disk icon in the same toolbar   -> vector SVG
  * two big buttons under the chart          -> PNG and SVG

Why this exists: server-side export (kaleido) crashes on Streamlit Cloud and
Render because their containers cannot start Chrome. Plotly.js can create
PNG and SVG files inside the browser itself, so that always works.
"""

import inspect
import json

import streamlit as st
from plotly.offline import get_plotlyjs_version


_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; padding: 0; background: transparent;
               font-family: Arial, Helvetica, sans-serif; }
  #plot { width: 100%; }
  #bar { display: flex; gap: 10px; align-items: center; padding: 8px 4px; }
  .dl { background: #0072B2; color: white; border: 0; border-radius: 6px;
        padding: 8px 16px; font-size: 14px; cursor: pointer; }
  .dl:hover { background: #005a8d; }
  #msg { font-size: 13px; color: #2e7d32; }
  #err { display: none; padding: 12px; border: 1px solid #d32f2f; color: #b71c1c;
         border-radius: 6px; font-size: 14px; background: #fdecea; }
</style>
</head>
<body>
<div id="err">
  The chart library could not be loaded (internet blocked or very slow).
  Tick <b>"Use basic viewer"</b> under the chart to see the figure and
  download a PNG instead.
</div>
<div id="plot"></div>
<div id="bar">
  <button class="dl" id="btn-png">&#11015; Download PNG (high-res)</button>
  <button class="dl" id="btn-svg">&#11015; Download SVG (vector)</button>
  <span id="msg"></span>
</div>

<script>
var FIG = __FIG_JSON__;
var EXPORT_W = __EXPORT_W__;
var EXPORT_H = __EXPORT_H__;
var FNAME = __FILENAME__;
var PNG_SCALE = __PNG_SCALE__;
var PRIMARY_SRC = __PRIMARY_SRC__;
var BACKUP_SRC = __BACKUP_SRC__;
var gd = null;

function say(text) {
  var m = document.getElementById("msg");
  m.textContent = text;
  setTimeout(function () { m.textContent = ""; }, 4000);
}

function download(fmt) {
  if (!gd || typeof Plotly === "undefined") { return; }
  var opts = { format: fmt, filename: FNAME, width: EXPORT_W, height: EXPORT_H };
  if (fmt === "png") { opts.scale = PNG_SCALE; }
  Plotly.downloadImage(gd, opts).then(function () {
    say(fmt.toUpperCase() + " download started");
  }).catch(function (e) {
    say("Export failed: " + e);
  });
}

function start() {
  gd = document.getElementById("plot");
  var layout = FIG.layout || {};
  // Show the chart at the width of the page; the export keeps EXPORT_W x EXPORT_H.
  delete layout.width;
  layout.autosize = true;
  var config = {
    responsive: true,
    displaylogo: false,
    // Privacy: Plotly.js 4 has a "Share chart..." button that uploads the
    // figure to Plotly Cloud. Researchers' data must never leave BioAI, so
    // the button is switched off three ways.
    showSendToCloud: false,
    modeBarButtonsToRemove: ["sendChartToCloud", "sendDataToCloud"],
    plotlyServerURL: "",
    toImageButtonOptions: {
      format: "png", filename: FNAME,
      width: EXPORT_W, height: EXPORT_H, scale: PNG_SCALE
    },
    modeBarButtonsToAdd: [{
      name: "downloadSvg",
      title: "Download as SVG (vector)",
      icon: Plotly.Icons.disk,
      click: function () { download("svg"); }
    }]
  };
  Plotly.newPlot(gd, FIG.data, layout, config);
}

function loadScript(src, onOk, onFail) {
  var s = document.createElement("script");
  s.src = src;
  s.onload = onOk;
  s.onerror = onFail;
  document.head.appendChild(s);
}

document.getElementById("btn-png").onclick = function () { download("png"); };
document.getElementById("btn-svg").onclick = function () { download("svg"); };

loadScript(PRIMARY_SRC, start, function () {
  loadScript(BACKUP_SRC, start, function () {
    document.getElementById("err").style.display = "block";
    document.getElementById("bar").style.display = "none";
  });
});
</script>
</body>
</html>
"""


def _js_literal(obj):
    """JSON text that is safe to paste inside a <script> block."""
    return json.dumps(obj).replace("<", "\\u003c")


def build_export_html(fig, filename="figure", export_width=None,
                      export_height=None, png_scale=3):
    """Returns (html_string, iframe_height_px). Pure Python, easy to test."""

    export_w = int(export_width or fig.layout.width or 1200)
    export_h = int(export_height or fig.layout.height or 800)
    display_h = int(fig.layout.height or 600)

    version = get_plotlyjs_version()
    primary = f"https://cdn.plot.ly/plotly-{version}.min.js"
    backup = f"https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{version}/plotly.min.js"

    fig_dict = json.loads(fig.to_json())

    html = (
        _HTML_TEMPLATE
        .replace("__FIG_JSON__", _js_literal(fig_dict))
        .replace("__EXPORT_W__", str(export_w))
        .replace("__EXPORT_H__", str(export_h))
        .replace("__FILENAME__", _js_literal(str(filename)))
        .replace("__PNG_SCALE__", str(float(png_scale)))
        .replace("__PRIMARY_SRC__", _js_literal(primary))
        .replace("__BACKUP_SRC__", _js_literal(backup))
    )

    return html, display_h + 80


def _show_html(html, height):
    """st.iframe on new Streamlit, components.html on older versions."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)


def _show_basic_chart(fig, key, filename, export_width, export_height, png_scale):
    """Plain Streamlit chart (PNG camera only). Safety net if the iframe fails."""
    config = {
        "toImageButtonOptions": {
            "format": "png", "filename": filename,
            "width": int(export_width or fig.layout.width or 1200),
            "height": int(export_height or fig.layout.height or 800),
            "scale": png_scale,
        },
        "displaylogo": False,
    }
    params = inspect.signature(st.plotly_chart).parameters
    if "width" in params:
        st.plotly_chart(fig, width="stretch", key=f"{key}_basic_chart", config=config)
    else:
        st.plotly_chart(fig, use_container_width=True,
                        key=f"{key}_basic_chart", config=config)


def render_exportable_plot(fig, key="figure", filename="figure",
                           export_width=None, export_height=None, png_scale=3):
    """
    Shows `fig` with PNG + SVG export. `key` must be unique per chart on a page.
    """

    use_basic = st.session_state.get(f"{key}_use_basic", False)

    if use_basic:
        _show_basic_chart(fig, key, filename, export_width, export_height, png_scale)
    else:
        html, height = build_export_html(
            fig, filename, export_width, export_height, png_scale
        )
        _show_html(html, height)

    st.checkbox(
        "Use basic viewer (only if the chart above does not appear)",
        key=f"{key}_use_basic",
    )
