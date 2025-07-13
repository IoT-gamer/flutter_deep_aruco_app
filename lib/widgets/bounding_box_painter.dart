import 'dart:math';
import 'package:flutter/material.dart';

import '../services/processing_service.dart';

class BoundingBoxPainter extends CustomPainter {
  final List<Rect> recognitions;
  final Size imageSize;
  final List<SimplePoint> detectedCorners;
  final int markerRotation;

  BoundingBoxPainter({
    required this.recognitions,
    required this.imageSize,
    required this.detectedCorners,
    required this.markerRotation,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.lightGreenAccent;

    if (recognitions.isEmpty) {
      return;
    }

    // Screen and Image dimensions
    final screenW = size.width;
    final screenH = size.height;
    final imageW = imageSize.width;
    final imageH = imageSize.height;

    // The camera stream provides images in landscape, but the preview widget is portrait.
    // This requires a 90-degree rotation in our coordinate mapping.
    // We calculate the scale factor to fit the landscape image within the portrait preview,
    // preserving its aspect ratio (letterboxing).
    final scaleX = screenW / imageH;
    final scaleY = screenH / imageW;
    final scale = min(scaleX, scaleY);

    // The size of the scaled, letterboxed image on the screen
    final scaledW = imageH * scale;
    final scaledH = imageW * scale;

    // The offset to center the letterboxed image
    final offsetX = (screenW - scaledW) / 2;
    final offsetY = (screenH - scaledH) / 2;

    // Extract the normalized bounding box from the YOLO model
    final normRect = recognitions.first;

    // Expand the box to match the input of the refiner model
    const double expansionFactor = 0.20;
    final double widthAdjustment = normRect.width * expansionFactor / 2.0;
    final double heightAdjustment = normRect.height * expansionFactor / 2.0;
    final expandedRect = Rect.fromLTRB(
      max(0.0, normRect.left - widthAdjustment),
      max(0.0, normRect.top - heightAdjustment),
      min(1.0, normRect.right + widthAdjustment),
      min(1.0, normRect.bottom + heightAdjustment),
    );

    // Map the normalized, expanded box coordinates to the screen,
    // applying the rotation, scaling, and offset.
    final screenRect = Rect.fromLTRB(
      (1.0 - expandedRect.bottom) * scaledW + offsetX, // Rotated Y -> Screen X
      expandedRect.left * scaledH + offsetY, // Rotated X -> Screen Y
      (1.0 - expandedRect.top) * scaledW + offsetX, // Rotated Y -> Screen X
      expandedRect.right * scaledH + offsetY, // Rotated X -> Screen Y
    );

    canvas.drawRect(screenRect, boxPaint);

    // Draw the detected corners if they exist
    if (detectedCorners.isNotEmpty) {
      _drawCorners(canvas, screenRect, detectedCorners);
    }
  }

  void _drawCorners(Canvas canvas, Rect screenRect, List<SimplePoint> corners) {
    final cornerPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.red;
    final bottomLeftPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.blueAccent;
    final double scaleX = screenRect.width / 64.0;
    final double scaleY = screenRect.height / 64.0;

    // This new formula accounts for the painter's coordinate rotation.
    final int bottomLeftIndex = (2 - markerRotation + 4) % 4;

    for (int i = 0; i < corners.length; i++) {
      final corner = corners[i];
      final screenX = screenRect.left + (63.0 - corner.y) * scaleX;
      final screenY = screenRect.top + corner.x * scaleY;
      final paint = (i == bottomLeftIndex) ? bottomLeftPaint : cornerPaint;
      canvas.drawCircle(Offset(screenX, screenY), 5.0, paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
