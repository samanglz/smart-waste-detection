1. purpose of this project

The purpose of this guideline is to ensure that all annotations are created consistently and accurately across the entire dataset.Following these rules will improve annotation quality and help produce a reliable object detection model.

# Annotation_Type
object detection 
shape:Bounding_box


2. labels
- cardboard
- glass
- metal
- paper
- plastic

Only these five classes are allowed. Any object that does not belong to one of these classes must not be annotated.

3. Bounding Box Rules
Rule 1 — Tight Bounding Boxes

Draw each bounding box as tightly as possible around the visible boundaries of the object.

Do not include unnecessary background pixels.

Rule 2 — Truncated Objects

If an object is partially outside the image, annotate only the visible portion.

Never estimate or annotate invisible regions.

Rule 3 — Occluded Objects

If two or more objects overlap, annotate only the visible part of each object.

Do not estimate hidden boundaries.

Rule 4 — Small Objects

Objects smaller than 10 pixels (width or height) must not be annotated.

Rule 5 — Reflections

Reflections in mirrors, glass, water, or other reflective surfaces must not be annotated.

Rule 6 — Uncertain Objects

If you cannot confidently determine which class an object belongs to, do not annotate it.

When in doubt, skip the object instead of guessing.

5. Label Assignment Rules
If an object has two labels annotate both 
Two separate objects must never share the same bounding box.
Every visible object belonging to one of the five classes should be annotated.
6. Truncated Objects

Objects cut off by the image boundary should still be annotated if their class can be identified.

Only annotate the visible part of the object.

7. Occluded Objects

Objects partially hidden behind other objects should be annotated if they can still be recognized.

Only annotate the visible portion.

If the object cannot be identified confidently, do not annotate it.

8. Small Objects

Very small objects that cannot be reliably recognized or are smaller than the minimum annotation size should be ignored.

9. Blurred Objects

Blurred objects should be annotated only if their class can still be confidently identified.

Otherwise, ignore them.

10. Correct Annotation Examples

(Images will be added later.)

Examples should demonstrate:

Tight bounding boxes
Proper handling of truncated objects
Proper handling of occluded objects
Correct class assignment
11. Incorrect Annotation Examples

(Images will be added later.)

Examples should demonstrate:

Loose bounding boxes
Wrong class labels
Missing objects
Multiple objects inside one bounding box
Annotating reflections
Annotating uncertain objects
Estimating invisible object boundaries
