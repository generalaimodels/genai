# Media Test - Images & Videos

## External Images

### Markdown Image
![Sample Image](https://picsum.photos/800/400)

### HTML Image
<img src="https://picsum.photos/600/300" alt="HTML Image Test" width="600" />

## Videos

### HTML5 Video
<video width="640" height="360" controls>
  <source src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Smaller Video
<video width="480" height="270" controls>
  <source src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4" type="video/mp4">
  Your browser does not support HTML5 video.
</video>

## Audio

<audio controls>
  <source src="https://commondatastorage.googleapis.com/codeskulptor-assets/Epoq-Lepidoptera.ogg" type="audio/ogg">
  Your browser does not support the audio element.
</audio>

## iFrame (YouTube)

<iframe width="560" height="315" src="https://www.youtube.com/embed/dQw4w9WgXcQ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Testing Notes

- All media should load and display correctly
- Videos should have controls
- Images should be responsive
- No console errors for CSP violations
