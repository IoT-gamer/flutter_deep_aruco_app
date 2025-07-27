import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'dart:typed_data';
import 'package:flutter/services.dart'
    show rootBundle; // Import for loading assets
import 'package:opencv_dart/opencv.dart' as cv;

import '../services/yolo_service.dart';
import '../services/processing_service.dart';
import '../widgets/bounding_box_painter.dart';
import '../widgets/debug_viewer.dart';

class DetectionScreen extends StatefulWidget {
  const DetectionScreen({super.key});
  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  CameraController? _cameraController;
  late YoloService _yoloService;
  late ProcessingService _processingService;
  late Future<void> _initFuture;
  List<Rect> _detectedBoxes = [];
  List<SimplePoint> _detectedCorners = [];
  bool _isDetecting = false;
  Size? _imageSize;
  int? _markerId;
  int _markerRotation = 0;
  Uint8List? _heatmapImage;
  Uint8List? _warpedImage;
  StreamSubscription? _heatmapSubscription;
  StreamSubscription? _warpedSubscription;
  // Debug viewer default to off for better performance
  bool _showDebugViewer = false;

  Uint8List? _overlayImageBytes; // To store the bytes of the image to overlay
  Uint8List? _overlayedFrameBytes; // To store the frame with the overlay
  bool _showOverlay = false; // To toggle the overlay display

  @override
  void initState() {
    super.initState();
    _initFuture = _initializeApp();
  }

  Future<void> _initializeApp() async {
    _yoloService = YoloService();
    _processingService = ProcessingService();

    await _yoloService.init();
    await _processingService.init();

    // Load the overlay image from assets
    try {
      _overlayImageBytes = (await rootBundle.load(
        'assets/overlay_image.png',
      )).buffer.asUint8List();
    } catch (e) {
      print('Error loading overlay image: $e');
      // Handle error, maybe use a default or show a message
    }

    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      throw Exception('No cameras found on this device.');
    }
    final controller = CameraController(
      cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
    );
    _cameraController = controller;

    await _cameraController!.initialize();
    await _cameraController!.startImageStream(_processImageStream);
  }

  Future<void> _processImageStream(CameraImage image) async {
    if (_isDetecting) return;
    _isDetecting = true;

    cv.Mat? fullFrameCvMat; // Declare here for proper disposal
    Uint8List? currentFrameBytesForDisplay; // This will be the image to show

    try {
      if (_imageSize == null && mounted) {
        setState(() {
          _imageSize = Size(image.width.toDouble(), image.height.toDouble());
        });
      }

      // Convert the CameraImage to cv.Mat (BGR format) for OpenCV operations
      fullFrameCvMat = _convertCameraImageToCvMat(image);
      final detectionResults = await _yoloService.predict(image);
      MarkerResult? markerResult;
      Uint8List? overlayedFrameFromService; // Holds the image from decodeMarker

      if (detectionResults.isNotEmpty) {
        final box = detectionResults.first;
        final refinerResult = await _processingService.refine(image, box);

        if (refinerResult != null) {
          final (croppedImageTensor, heatmap) = refinerResult;
          final corners = await _processingService.findCorners(heatmap);

          if (corners.length == 4 && _overlayImageBytes != null) {
            // Pass the full frame Mat, overlay image bytes, and the original detection box
            final (mr, overlayBytes) = await _processingService.decodeMarker(
              // Modified to get result and image bytes
              fullFrameCvMat,
              croppedImageTensor,
              corners,
              _overlayImageBytes!, // Pass the loaded overlay image bytes
              box, // Pass the original detection box
            );
            markerResult = mr; // Assign MarkerResult
            overlayedFrameFromService =
                overlayBytes; // Assign overlayed frame bytes
          }
        }
      }

      // This logic ensures that if a marker is found, its data is used.
      // If it's not found for any reason, the UI is cleared of old results.
      if (mounted) {
        setState(() {
          if (markerResult != null) {
            _detectedBoxes = detectionResults;
            // Use the canonically sorted corners from the result for drawing
            _detectedCorners = markerResult!.corners;
            _markerId = markerResult.id;
            _markerRotation = markerResult.rotation;

            if (_showOverlay && overlayedFrameFromService != null) {
              cv.Mat tempMat = cv.imdecode(
                overlayedFrameFromService,
                cv.IMREAD_COLOR,
              );
              cv.Mat rotatedDisplayMat;
              // Determine if rotation is needed based on image aspect ratio
              // Assuming your camera provides landscape images and your app is portrait
              if (tempMat.cols > tempMat.rows) {
                // If image is landscape
                rotatedDisplayMat = cv.rotate(
                  tempMat,
                  cv.ROTATE_90_CLOCKWISE,
                ); // Rotate for portrait display
              } else {
                rotatedDisplayMat = tempMat
                    .clone(); // Already portrait or square
              }
              currentFrameBytesForDisplay = cv
                  .imencode('.jpg', rotatedDisplayMat)
                  .$2;
              tempMat.dispose();
              rotatedDisplayMat.dispose();
            } else {
              // If no overlay, display the raw camera frame (also needs rotation if it's landscape)
              cv.Mat rawMat = fullFrameCvMat!
                  .clone(); // Clone to rotate without affecting original `fullFrameCvMat` for `decodeMarker`
              cv.Mat rotatedRawMat;
              if (rawMat.cols > rawMat.rows) {
                rotatedRawMat = cv.rotate(rawMat, cv.ROTATE_90_CLOCKWISE);
              } else {
                rotatedRawMat = rawMat.clone();
              }
              currentFrameBytesForDisplay = cv
                  .imencode('.jpg', rotatedRawMat)
                  .$2;
              rawMat.dispose();
              rotatedRawMat.dispose();
            }
          } else {
            // Clear previous results if no marker is found in this frame
            _detectedBoxes = [];
            _detectedCorners = [];
            _markerId = null;
            _markerRotation = 0;
            _overlayedFrameBytes = null; // Clear if no marker found
            // Also handle raw camera frame rotation if no marker found
            cv.Mat rawMat = fullFrameCvMat!.clone();
            cv.Mat rotatedRawMat;
            if (rawMat.cols > rawMat.rows) {
              rotatedRawMat = cv.rotate(rawMat, cv.ROTATE_90_CLOCKWISE);
            } else {
              rotatedRawMat = rawMat.clone();
            }
            currentFrameBytesForDisplay = cv.imencode('.jpg', rotatedRawMat).$2;
            rawMat.dispose();
            rotatedRawMat.dispose();
          }
        });
      }
    } catch (e, stackTrace) {
      print('[SCREEN] Error in image processing stream: $e');
      print('[SCREEN] Stack trace: $stackTrace');
      if (mounted) {
        setState(() {
          _overlayedFrameBytes = null; // Clear overlay on error
          currentFrameBytesForDisplay = fullFrameCvMat != null
              ? cv.imencode('.jpg', fullFrameCvMat).$2
              : null; // Fallback on error
        });
      }
    } finally {
      fullFrameCvMat?.dispose(); // Ensure the Mat is disposed
      _isDetecting = false;
      // Update the UI with the chosen frame (raw or overlayed)
      if (mounted && currentFrameBytesForDisplay != null) {
        setState(() {
          _overlayedFrameBytes =
              currentFrameBytesForDisplay; // Final update for display
        });
      }
    }
  }

  /// Converts a CameraImage (YUV420_888) to an OpenCV Mat (BGR).
  /// This function runs on the main isolate.
  cv.Mat _convertCameraImageToCvMat(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final Uint8List yPlane = image.planes[0].bytes;
    final Uint8List uPlane = image.planes[1].bytes;
    final Uint8List vPlane = image.planes[2].bytes;
    final int yRowStride = image.planes[0].bytesPerRow;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel!;
    // Create a list to hold BGR data
    final bgrBytes = Uint8List(width * height * 3);
    int bgrIndex = 0;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex = uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
        final int yIndex = y * yRowStride + x;

        final yp = yPlane[yIndex];
        final up = uPlane[uvIndex];
        final vp = vPlane[uvIndex];

        // YUV to BGR conversion
        // Source for conversion: https://en.wikipedia.org/wiki/YUV#Y%E2%80%B2UV420p_(and_Y%E2%80%B2V12_or_YV12)
        // Using integer arithmetic for speed.
        // Y values are 0-255, U/V are -128 to 127, but in bytes they are 0-255 (shifted by 128)
        final C = yp - 16;
        final D = up - 128;
        final E = vp - 128;
        int b = ((298 * C + 516 * D + 128) >> 8).clamp(0, 255);
        int g = ((298 * C - 100 * D - 208 * E + 128) >> 8).clamp(0, 255);
        int r = ((298 * C + 409 * E + 128) >> 8).clamp(0, 255);

        bgrBytes[bgrIndex++] = b;
        bgrBytes[bgrIndex++] = g;
        bgrBytes[bgrIndex++] = r;
      }
    }
    // Create a CV_8UC3 Mat from the BGR bytes
    return cv.Mat.fromList(height, width, cv.MatType.CV_8UC3, bgrBytes);
  }

  @override
  void dispose() {
    _heatmapSubscription?.cancel();
    _warpedSubscription?.cancel();
    _cameraController?.dispose();
    _yoloService.dispose();
    _processingService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: FutureBuilder<void>(
        future: _initFuture,
        builder: (context, snapshot) {
          if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          }
          if (snapshot.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }
          final controller = _cameraController;
          if (controller == null) {
            return const Center(child: Text('Camera not available.'));
          }
          final previewSize = controller.value.previewSize;
          if (previewSize == null || _imageSize == null) {
            return const Center(child: Text('Initializing...'));
          }
          return Center(
            child: AspectRatio(
              aspectRatio: previewSize.height / previewSize.width,
              child: Stack(
                children: [
                  // Display the frame: either original or overlayed
                  if (_overlayedFrameBytes != null)
                    Image.memory(
                      _overlayedFrameBytes!,
                      fit: BoxFit.cover,
                      width: double.infinity,
                      height: double.infinity,
                      gaplessPlayback: true, // Important to prevent flickering
                    )
                  else
                    CameraPreview(
                      controller,
                    ), // Fallback to CameraPreview if no overlay is set

                  CustomPaint(
                    size: Size.infinite,
                    painter: BoundingBoxPainter(
                      recognitions: _detectedBoxes,
                      imageSize: _imageSize!,
                      detectedCorners: _detectedCorners,
                      markerRotation: _markerRotation,
                    ),
                  ),
                  if (_showDebugViewer)
                    DebugViewer(heatmap: _heatmapImage, warped: _warpedImage),
                  if (_markerId != null)
                    Positioned(
                      top: 0,
                      left: 20,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 8,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.black54,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          'ID: $_markerId',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ),
                  Positioned(
                    // Original debug toggle button
                    bottom: 30,
                    right: 30,
                    child: FloatingActionButton(
                      heroTag: "debugViewer", // Add a unique tag
                      backgroundColor: Colors.black.withValues(
                        alpha: 0.5,
                      ), // Corrected opacity
                      onPressed: () {
                        setState(() {
                          _showDebugViewer = !_showDebugViewer;
                          if (_showDebugViewer) {
                            // Start listening to the streams
                            _heatmapSubscription = _processingService
                                .heatmapStream
                                .listen((image) {
                                  if (mounted) {
                                    setState(() => _heatmapImage = image);
                                  }
                                });
                            _warpedSubscription = _processingService
                                .warpedStream
                                .listen((image) {
                                  if (mounted) {
                                    setState(() => _warpedImage = image);
                                  }
                                });
                          } else {
                            // Stop listening and clear images
                            _heatmapSubscription?.cancel();
                            _warpedSubscription?.cancel();
                            _heatmapImage = null;
                            _warpedImage = null;
                          }
                        });
                      },
                      child: Icon(
                        _showDebugViewer
                            ? Icons.visibility_off
                            : Icons.visibility,
                        color: Colors.white,
                      ),
                    ),
                  ),
                  // Overlay toggle button
                  Positioned(
                    bottom: 30,
                    left: 30,
                    child: FloatingActionButton(
                      heroTag: "overlayToggle", // Add a unique tag
                      backgroundColor: Colors.blue.withValues(
                        alpha: 0.5,
                      ), // Different color for distinction
                      onPressed: () {
                        setState(() {
                          _showOverlay =
                              !_showOverlay; // Toggle the overlay visibility state
                          // When toggling, force a redraw by potentially clearing the _overlayedFrameBytes
                          if (!_showOverlay) {
                            _overlayedFrameBytes = null; // Clear if turning off
                          }
                        });
                      },
                      child: Icon(
                        _showOverlay ? Icons.visibility_off : Icons.image,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
