# Contributing guidelines

## Markup quirks

- Lines cannot have trailing spaces

- Do not combine markdown images, i.e., `![caption](image.png)`,
  with our caption cruncher.  The "alt" text ("caption" in this
  example) is used as the image caption by default, or you can combine
  `<img>` tags with `<!-- caption text="caption" -->` tags.
