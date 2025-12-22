
# FFmpeg: Convert RGB video to true grayscale

```shell
ffmpeg -i input.mkv -vf format=gray -pix_fmt gray output.mkv
```

-`vf format=gray` → tells FFmpeg to filter video into gray.
-`pix_fmt gray` → sets the pixel format to one channel.

# 1. How values change when converting RGB → Grayscale
Humans see green as brightest, then red, then blue as darkest.
`Gray = 0.299*R + 0.587*G + 0.114*B`
This is what FFmpeg (and most software) uses by default when converting RGB to grayscale.


