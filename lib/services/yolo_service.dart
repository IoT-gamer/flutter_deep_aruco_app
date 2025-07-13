// lib/services/yolo_service.dart
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:integral_isolates/integral_isolates.dart';
import 'package:ultralytics_yolo/yolo.dart';

// 1. Create a data class to safely pass CameraImage data to the isolate.
//    This is identical to the pattern in your hand_landmarker.dart example.
class _IsolateData {
  final Uint8List yPlane;
  final Uint8List uPlane;
  final Uint8List vPlane;
  final int yRowStride; // ADD THIS
  final int uvRowStride;
  final int uvPixelStride;
  final int width;
  final int height;

  _IsolateData(CameraImage image)
    : yPlane = image.planes[0].bytes,
      yRowStride = image.planes[0].bytesPerRow, // ADD THIS
      uPlane = image.planes[1].bytes,
      vPlane = image.planes[2].bytes,
      uvRowStride = image.planes[1].bytesPerRow,
      uvPixelStride = image.planes[1].bytesPerPixel!,
      height = image.height,
      width = image.width;
}

// 2. Move the conversion function to be a top-level function so the isolate can access it.
Future<Uint8List?> _convertYuvToJpg(_IsolateData isolateData) async {
  try {
    final image = img.Image(
      width: isolateData.width,
      height: isolateData.height,
    );
    final yRowStride = isolateData.yRowStride; // Get stride
    final planeY = isolateData.yPlane;
    final planeU = isolateData.uPlane;
    final planeV = isolateData.vPlane;
    final uvRowStride = isolateData.uvRowStride;
    final uvPixelStride = isolateData.uvPixelStride;

    for (int y = 0; y < isolateData.height; y++) {
      for (int x = 0; x < isolateData.width; x++) {
        final int uvIndex =
            isolateData.uvPixelStride * (x / 2).floor() +
            isolateData.uvRowStride * (y / 2).floor();
        final int index = y * yRowStride + x;

        final yp = planeY[index];
        final up = isolateData.uPlane[uvIndex];
        final vp = isolateData.vPlane[uvIndex];

        int r = (yp + vp * 1436 / 1024 - 179).round();
        int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
            .round();
        int b = (yp + up * 1814 / 1024 - 227).round();

        image.setPixelRgb(
          x,
          y,
          r.clamp(0, 255),
          g.clamp(0, 255),
          b.clamp(0, 255),
        );
      }
    }
    return Uint8List.fromList(img.encodeJpg(image));
  } catch (e) {
    print("Error converting YUV to JPG in isolate: $e");
    return null;
  }
}

class YoloService {
  late final YOLO yolo;
  // 3. Create a handle for our stateful isolate.
  late final StatefulIsolate _isolate;

  Future<void> init() async {
    // 4. Initialize the isolate. We use a backpressure strategy to handle
    //    cases where frames come in faster than they can be processed.
    _isolate = StatefulIsolate(
      backpressureStrategy: ReplaceBackpressureStrategy(),
    );

    // For Android, YOLO model is in android/app/src/main/assets/
    yolo = YOLO(modelPath: 'detector_int8.tflite', task: YOLOTask.detect);
    await yolo.loadModel();
    print('YOLO TFLite model loaded successfully.');
  }

  Future<List<Rect>> predict(CameraImage image) async {
    // 5. Delegate the heavy conversion work to the isolate.
    final jpgBytes = await _isolate.compute(
      _convertYuvToJpg,
      _IsolateData(image),
    );

    if (jpgBytes == null) {
      return [];
    }

    final Map<String, dynamic> results = await yolo.predict(
      jpgBytes,
      confidenceThreshold: 0.5,
    );
    final List<dynamic> boxes = results['boxes'] as List<dynamic>? ?? [];

    if (boxes.isEmpty) {
      return [];
    }

    return boxes.map((box) {
      final boxMap = box as Map<String, dynamic>;
      final double left = (boxMap['x1_norm'] as num?)?.toDouble() ?? 0.0;
      final double top = (boxMap['y1_norm'] as num?)?.toDouble() ?? 0.0;
      final double right = (boxMap['x2_norm'] as num?)?.toDouble() ?? 0.0;
      final double bottom = (boxMap['y2_norm'] as num?)?.toDouble() ?? 0.0;
      return Rect.fromLTRB(left, top, right, bottom);
    }).toList();
  }

  void dispose() {
    _isolate.dispose();
    yolo.dispose();
  }
}
