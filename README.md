# Method of Moving Asymptotes

> [!NOTE]
> This repository is forked from
> [arjendeetman/GCMMA-MMA-Python](https://github.com/arjendeetman/GCMMA-MMA-Python)
> for future experimental changes and refactoring. Please use the original
> rather than this fork if you need to rely on this MMA implementation.

This repository implements a variation of the Method of Moving Asymptotes by
[Svanberg, 1987](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620240207),
originally implemented in MATLAB, translated by
[arjendeetman](https://github.com/arjendeetman) and then forked. Since the
original was provided under the GNU General Public License the current version
remains licensed as such.

## Installation and usage

```bash
uv sync
uv run pytest
```

## References

* Svanberg, K. (1987). The Method of Moving Asymptotes – A new method for
  structural optimization. International Journal for Numerical Methods in
  Engineering 24, 359-373.
  [doi:10.1002/nme.1620240207](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620240207)
* Svanberg, K. (n.d.). MMA and GCMMA – two methods for nonlinear optimization.
  Retrieved August 3, 2017 from  https://people.kth.se/~krille/mmagcmma.pdf

## License

All versions are provided under the [GPL-3.0 license](LICENSE) with copyright:

* Original work written in MATLAB: Copyright (c) 2006-2009 Krister Svanberg\
* Derived Python implementation: Copyright (c) 2020-2024 Arjen Deetman
* Derived Python implementation: Copyright (c) 2024-current cont00rs
