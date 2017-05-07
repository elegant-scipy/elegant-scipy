# Contributing guidelines

## Markup quirks

- Lines cannot start with "$", e.g.

  ```
  $X$ is a matrix
  ```

  won't compile, so move the "$X" to the end of the previous line.

- Lines cannot have trailing spaces

- Use `<img>` tags directly, instead of the markdown syntax:
  `![caption](image.png)`
