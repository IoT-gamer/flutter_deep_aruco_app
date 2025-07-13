import 'dart:async';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:integral_isolates/integral_isolates.dart';
import 'package:opencv_dart/opencv.dart' as cv;
import 'package:tflite_flutter/tflite_flutter.dart';

class SimplePoint {
  final int x;
  final int y;
  SimplePoint(this.x, this.y);
}

class MarkerResult {
  final int id;
  final int rotation;
  final List<SimplePoint> corners;

  MarkerResult(this.id, this.rotation, this.corners);
}

// This data class will be passed to the isolate function
class _CropIsolateData {
  final CameraImage image;
  final Rect box;

  _CropIsolateData(this.image, this.box);
}

/// This top-level function runs on the isolate.
/// It crops the region of interest directly from the YUV planes and then converts
/// only the cropped section to a resized RGB Float32List.
Float32List _cropAndResize(_CropIsolateData isolateData) {
  final image = isolateData.image;
  final box = isolateData.box;
  const int targetSize = 64; // The refiner model's input size

  final imageWidth = image.width;
  final imageHeight = image.height;

  // 1. Expand the box to get more context for the refiner model
  const double expansionFactor = 0.20;
  final double widthAdjustment = box.width * expansionFactor / 2.0;
  final double heightAdjustment = box.height * expansionFactor / 2.0;
  final expandedBox = Rect.fromLTRB(
    max(0.0, box.left - widthAdjustment),
    max(0.0, box.top - heightAdjustment),
    min(1.0, box.right + widthAdjustment),
    min(1.0, box.bottom + heightAdjustment),
  );

  // 2. Calculate pixel coordinates for the crop area
  final int cropLeft = (expandedBox.left * imageWidth).toInt();
  final int cropTop = (expandedBox.top * imageHeight).toInt();
  final int cropWidth = (expandedBox.width * imageWidth).toInt();
  final int cropHeight = (expandedBox.height * imageHeight).toInt();

  // 3. Get YUV plane data and strides
  final yPlane = image.planes[0].bytes;
  final uPlane = image.planes[1].bytes;
  final vPlane = image.planes[2].bytes;
  final yStride = image.planes[0].bytesPerRow;
  final uvStride = image.planes[1].bytesPerRow;
  final uvPixelStride = image.planes[1].bytesPerPixel!;

  // 4. Create the Float32List to hold the final RGB data for the model
  final imageAsList = Float32List(targetSize * targetSize * 3);
  int listIndex = 0;

  // 5. Iterate through the target 64x64 grid
  for (int y = 0; y < targetSize; y++) {
    for (int x = 0; x < targetSize; x++) {
      // Map the 64x64 grid coordinate to the original image's crop area coordinate
      final int originalX = cropLeft + (x * cropWidth / targetSize).toInt();
      final int originalY = cropTop + (y * cropHeight / targetSize).toInt();

      // Get the YUV values for the pixel in the original image
      final yIndex = originalY * yStride + originalX;
      final uvIndex =
          (originalY / 2).floor() * uvStride +
          (originalX / 2).floor() * uvPixelStride;

      final yp = yPlane[yIndex];
      final up = uPlane[uvIndex];
      final vp = vPlane[uvIndex];

      // YUV to RGB conversion
      double r = (yp + vp * 1436 / 1024 - 179);
      double g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91);
      double b = (yp + up * 1814 / 1024 - 227);

      // Normalize and add to the list
      imageAsList[listIndex++] = r.clamp(0, 255) / 255.0;
      imageAsList[listIndex++] = g.clamp(0, 255) / 255.0;
      imageAsList[listIndex++] = b.clamp(0, 255) / 255.0;
    }
  }

  return imageAsList;
}

class ProcessingService {
  late final Interpreter _refiner;
  late final StatefulIsolate _isolate;
  // late final cv.SimpleBlobDetector _blobDetector;

  late final cv.ArucoDictionary _arucoDictionary;

  // Add controllers to broadcast the debug images
  final _heatmapStreamController = StreamController<Uint8List>.broadcast();
  final _warpedStreamController = StreamController<Uint8List>.broadcast();
  // Expose the streams for the UI to listen to
  Stream<Uint8List> get heatmapStream => _heatmapStreamController.stream;
  Stream<Uint8List> get warpedStream => _warpedStreamController.stream;

  Future<void> init() async {
    _refiner = await Interpreter.fromAsset('assets/refiner32.tflite');
    _isolate = StatefulIsolate(
      backpressureStrategy: ReplaceBackpressureStrategy(),
    );

    _arucoDictionary = cv.ArucoDictionary.predefined(
      cv.PredefinedDictionaryType.DICT_6X6_250,
    );
    // Create the blob detector. These parameters can be tuned.
    // // Create a parameters object for the blob detector.
    // final params = cv.SimpleBlobDetectorParams(
    //   // Add the required thresholding parameters
    //   minThreshold: 50.0,
    //   maxThreshold: 255.0,
    //   thresholdStep: 10.0,
    //   minDistBetweenBlobs: 10.0,
    //   minRepeatability: 5,

    //   // The corners are bright spots, so we look for a blob color of 255 (white).
    //   filterByColor: true,
    //   blobColor: 255,

    //   // Filter by area to find reasonably sized corner blobs.
    //   filterByArea: true,
    //   minArea: 4.0, // Adjust these values as needed
    //   maxArea: 100.0,

    //   // Filter by circularity. Corners should be somewhat circular.
    //   filterByCircularity: true,
    //   minCircularity: 0.2,
    //   maxCircularity: 1.0,

    //   // Other filters can be disabled for now.
    //   filterByConvexity: false,
    //   minConvexity: 0.2,
    //   maxConvexity: 1.0,
    //   filterByInertia: false,
    //   minInertiaRatio: 0.5,
    //   maxInertiaRatio: 1.0,
    // );
    // // Create the detector with the specified parameters.
    // _blobDetector = cv.SimpleBlobDetector.create(params);

    print('Processing Service initialized.');
  }

  /// Takes the full image and detection box, returns the refiner model's input
  /// AND its output heatmap.
  Future<(Float32List, dynamic)?> refine(
    CameraImage image,
    Rect detectionBox,
  ) async {
    try {
      // This is the cropped and resized image tensor
      final inputTensor = await _isolate.compute(
        _cropAndResize,
        _CropIsolateData(image, detectionBox),
      );
      if (inputTensor.isEmpty) return null;

      final reshapedInput = inputTensor.reshape([1, 64, 64, 3]);
      final output = List.filled(1 * 64 * 64 * 1, 0.0).reshape([1, 64, 64, 1]);
      _refiner.run(reshapedInput, output);

      // Return BOTH the input tensor and the output heatmap
      return (inputTensor, output);
    } catch (e) {
      print('Error in ProcessingService.refine: $e');
      return null;
    }
  }

  /// Method to find corners from the refiner's output heatmap with sub-pixel precision.
  /// It returns a list of Point objects.
  Future<List<SimplePoint>> findCorners(dynamic refinerOutput) async {
    cv.Mat? heatmapFloat;
    cv.Mat? heatmapByte;
    cv.Mat? thresholded;
    cv.VecVecPoint? contours;
    cv.Point2f? point2f;

    try {
      final List<double> outputList = (refinerOutput[0] as List)
          .expand<double>(
            (row) => (row as List).map<double>(
              (pixel) => (pixel[0] as num).toDouble(),
            ),
          )
          .toList();
      heatmapFloat = cv.Mat.fromList(64, 64, cv.MatType.CV_32FC1, outputList);

      heatmapByte = heatmapFloat.convertTo(cv.MatType.CV_8UC1, alpha: 255);

      // Stream heatmap
      // imencode returns a record (bool success, Uint8List data). We need the data part (.$2).
      final Uint8List encodedHeatmap = cv.imencode('.jpg', heatmapByte).$2;

      // Add the encoded heatmap bytes to the stream
      _heatmapStreamController.add(encodedHeatmap);

      // Lower the threshold to capture more of the heatmap blob.
      thresholded = cv
          .threshold(heatmapByte, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
          .$2;

      contours = cv
          .findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
          .$1;

      final contourList = contours.toList();
      contourList.sort(
        (a, b) => cv.contourArea(b).compareTo(cv.contourArea(a)),
      );
      final topContours = contourList.take(4).toList();

      final List<SimplePoint> refinedCorners = [];

      for (final contour in topContours) {
        final rect = cv.boundingRect(contour);
        double weightedX = 0;
        double weightedY = 0;
        double totalWeight = 0;

        for (int y = rect.y; y < rect.y + rect.height; y++) {
          for (int x = rect.x; x < rect.x + rect.width; x++) {
            point2f = cv.Point2f(x.toDouble(), y.toDouble());
            if (cv.pointPolygonTest(contour, point2f, false) >= 0) {
              final weight = heatmapFloat.at<double>(y, x);
              weightedX += x * weight;
              weightedY += y * weight;
              totalWeight += weight;
            }
            point2f.dispose();
            point2f = null;
          }
        }

        if (totalWeight > 0) {
          refinedCorners.add(
            SimplePoint(
              (weightedX / totalWeight).toInt(),
              (weightedY / totalWeight).toInt(),
            ),
          );
        }
      }

      return refinedCorners;
    } catch (e) {
      print('Error in ProcessingService.findCorners: $e');
      return [];
    } finally {
      heatmapFloat?.dispose();
      heatmapByte?.dispose();
      thresholded?.dispose();
      contours?.dispose();
      point2f?.dispose();
    }
  }

  Future<MarkerResult?> decodeMarker(
    Float32List croppedImageTensor,
    List<SimplePoint> corners,
  ) async {
    if (corners.length != 4) return null;
    cv.Mat? croppedMat;
    cv.VecPoint? srcPoints;
    cv.VecPoint? dstPoints;
    cv.Mat? transform;
    cv.Mat? warped;
    cv.Mat? warpedGray;
    cv.Mat? warpedGray8bit;
    cv.Mat? binaryMarker;
    cv.Mat? bits;

    try {
      List<SimplePoint> cornersCopy = List.from(corners);

      // Top-left has the smallest sum of x+y
      // Bottom-right has the largest sum of x+y
      cornersCopy.sort((a, b) => (a.x + a.y).compareTo(b.x + b.y));
      final topLeft = cornersCopy.first;
      final bottomRight = cornersCopy.last;

      // Remove these two to isolate the other two corners
      cornersCopy.remove(topLeft);
      cornersCopy.remove(bottomRight);

      // Of the remaining two, Top-Right has a smaller y-x difference
      // and Bottom-Left has a larger y-x difference.
      final topRight =
          (cornersCopy.first.y - cornersCopy.first.x) <
              (cornersCopy.last.y - cornersCopy.last.x)
          ? cornersCopy.first
          : cornersCopy.last;
      final bottomLeft =
          (cornersCopy.first.y - cornersCopy.first.x) <
              (cornersCopy.last.y - cornersCopy.last.x)
          ? cornersCopy.last
          : cornersCopy.first;

      // The canonical corner order for OpenCV ArUco
      final List<SimplePoint> canonicalCorners = [
        topLeft,
        topRight,
        bottomRight,
        bottomLeft,
      ];

      // The perspectiveCorners list from the previous step is still needed
      // to keep the warped image in the debug view upright.
      final List<SimplePoint> perspectiveCorners = [
        canonicalCorners[3], // Bottom-Left
        canonicalCorners[0], // Top-Left
        canonicalCorners[1], // Top-Right
        canonicalCorners[2], // Bottom-Right
      ];

      croppedMat = cv.Mat.fromList(
        64,
        64,
        cv.MatType.CV_32FC3,
        croppedImageTensor,
      );

      final srcPointsList = perspectiveCorners
          .map((p) => cv.Point(p.x, p.y))
          .toList();
      srcPoints = cv.VecPoint.fromList(srcPointsList);

      const int warpedSize = 80;
      final dstPointsList = [
        cv.Point(0, 0),
        cv.Point(warpedSize - 1, 0),
        cv.Point(warpedSize - 1, warpedSize - 1),
        cv.Point(0, warpedSize - 1),
      ];
      dstPoints = cv.VecPoint.fromList(dstPointsList);
      transform = cv.getPerspectiveTransform(srcPoints, dstPoints);
      warped = cv.warpPerspective(croppedMat, transform, (
        warpedSize,
        warpedSize,
      ));

      warpedGray = cv.cvtColor(warped, cv.COLOR_RGB2GRAY);
      warpedGray8bit = warpedGray.convertTo(cv.MatType.CV_8UC1, alpha: 255);
      binaryMarker = cv
          .threshold(warpedGray8bit, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
          .$2;
      final encodedWarpedBinary = cv.imencode('.jpg', binaryMarker).$2;
      _warpedStreamController.add(encodedWarpedBinary);

      const gridSize = 6;
      const borderSize = 1;
      const totalBlocks = gridSize + 2 * borderSize;
      final cellSize = warpedSize / totalBlocks;
      bits = cv.Mat.zeros(gridSize, gridSize, cv.MatType.CV_8UC1);
      for (int y = 0; y < gridSize; y++) {
        for (int x = 0; x < gridSize; x++) {
          final startX = ((x + borderSize) * cellSize).toInt();
          final startY = ((y + borderSize) * cellSize).toInt();
          final cell = binaryMarker.region(
            cv.Rect(startX, startY, cellSize.toInt(), cellSize.toInt()),
          );
          if (cv.countNonZero(cell) > (cellSize * cellSize) / 2) {
            bits.set<int>(y, x, 1);
          }
          cell.dispose();
        }
      }

      final (bool found, int id, int rotation) = _arucoDictionary.identify(
        bits,
        2.0,
      );

      if (found) {
        return MarkerResult(id, rotation, canonicalCorners);
      }
      return null;
    } catch (e) {
      print('Error in ProcessingService.decodeMarker: $e');
      return null;
    } finally {
      croppedMat?.dispose();
      srcPoints?.dispose();
      dstPoints?.dispose();
      transform?.dispose();
      warped?.dispose();
      warpedGray?.dispose();
      warpedGray8bit?.dispose();
      binaryMarker?.dispose();
      bits?.dispose();
    }
  }

  void dispose() {
    _refiner.close();
    _arucoDictionary.dispose();
    _isolate.dispose();
    _heatmapStreamController.close();
    _warpedStreamController.close();
  }
}
