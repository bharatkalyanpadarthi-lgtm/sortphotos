# Smart Album Structure

Smart albums are generated hardlink views under each person folder:

```text
photos_by_person/<person>/_smart_albums/
```

They do not move or duplicate originals. The real files stay in:

```text
photos_by_person/<person>/photos/
photos_by_person/<person>/photos_nude/
```

## Folder Meanings

- `00_best/` - top technical-quality picks by sharpness, resolution, exposure, and contrast.
- `01_quality/` - quality buckets such as high score, low resolution, low sharpness, and exposure issues.
- `02_format/` - simple image orientation: portrait, landscape, or square.
- `03_face_framing/` - AI-video/reference-use folders based on detected face angle, face size, centering, eye line, and roll.
- `04_visual_similar/` - near-duplicate or very similar-looking images using pHash plus color/geometry checks.
- `05_same_scene/` - broader scene/context groups using color, layout, face position, framing, and clustering.
- `07_review_needed/` - technical review buckets, currently mainly low-resolution images.
- `08_outfit/` - local heuristic outfit views using lower/central body-region color, shot framing, and face geometry.

## Face Framing

`03_face_framing/90_not_ideal_for_ai_video/` replaces the older `90_review/` name.

It means the image may still be useful, but it is not ideal as a clean AI-video reference. Typical reasons:

- contact sheet or collage
- face detection uncertain
- multi-face or group photo
- side profile or extreme angle
- off-center subject
- tilted or strongly rotated face
- needs manual angle review

The best straight-face folder is:

```text
03_face_framing/00_clear_straight_face_best/
```

That folder requires a front-facing face, decent centering, eye-level-ish placement, acceptable quality, and low roll.

## Outfit Folders

`08_outfit/` is intentionally labeled as "likely" because no heavy vision-language model is installed. It uses local OpenCV features only.

Current outfit views:

- `08_outfit/01_likely_saree_or_draped_ethnic/`
- `08_outfit/03_likely_western_or_modern/`
- `08_outfit/90_outfit_uncertain/`
- `08_outfit/00_visibility/outfit_visible/`
- `08_outfit/00_visibility/partial_outfit_visible/`
- `08_outfit/00_visibility/outfit_unclear_closeup/`
- `08_outfit/10_by_outfit_color/<color>/`
- `08_outfit/11_colorful_outfits/`
- `08_outfit/12_neutral_or_plain_outfits/`

The saree folder is a broad candidate set, not a guaranteed classifier. It favors images with visible body framing, saturated/rich garment color, and body-region cues that often appear in saree or draped ethnic photos.

## Nudity Inside Albums

There is no separate `06_nudity/` smart-album section anymore. Nude or possible-nude images are nested inside the useful album where they belong:

```text
05_same_scene/.../_nudity_possible/
04_visual_similar/.../_nudity_possible/
08_outfit/.../_nudity_possible/
03_face_framing/.../_nudity_possible/
```

This keeps scene, outfit, quality, and framing folders useful while still separating nude images inside each folder.

Automatic nudity movement is disabled in the normal scan and daily flow. New images stay in each person's normal `photos/` folder unless you explicitly run `python face.py nudity` for a separate review pass.

## Why Visual Similar And Same Scene Are Separate

`04_visual_similar/` is stricter. It is best for near duplicates and same-looking frames.

`05_same_scene/` is broader. It can group photos from the same shoot, background, lighting, outfit, or context even when the pose or crop changes.

Keeping both is useful:

- use `04_visual_similar/` to clean repeated images
- use `05_same_scene/` to browse shoot/scene variations
