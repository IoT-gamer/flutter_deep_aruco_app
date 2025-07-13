import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'dart:typed_data';

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

    try {
      if (_imageSize == null && mounted) {
        setState(() {
          _imageSize = Size(image.width.toDouble(), image.height.toDouble());
        });
      }

      final detectionResults = await _yoloService.predict(image);
      MarkerResult? markerResult;

      if (detectionResults.isNotEmpty) {
        final box = detectionResults.first;
        final refinerResult = await _processingService.refine(image, box);

        if (refinerResult != null) {
          final (croppedImageTensor, heatmap) = refinerResult;
          final corners = await _processingService.findCorners(heatmap);

          if (corners.length == 4) {
            // The decodeMarker function now returns a MarkerResult object
            markerResult = await _processingService.decodeMarker(
              croppedImageTensor,
              corners,
            );
          }
        }
      }

      // This logic ensures that if a marker is found, its data is used.
      // If it's not found for any reason, the UI is cleared of old results.
      if (mounted) {
        if (markerResult != null) {
          setState(() {
            _detectedBoxes = detectionResults;
            // Use the canonically sorted corners from the result for drawing
            _detectedCorners = markerResult!.corners;
            _markerId = markerResult.id;
            _markerRotation = markerResult.rotation;
          });
        } else {
          // Clear previous results if no marker is found in this frame
          setState(() {
            _detectedBoxes = [];
            _detectedCorners = [];
            _markerId = null;
            _markerRotation = 0;
          });
        }
      }
    } catch (e, stackTrace) {
      print('[SCREEN] Error in image processing stream: $e');
      print('[SCREEN] Stack trace: $stackTrace');
    }

    _isDetecting = false;
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
                  CameraPreview(controller),
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
                    bottom: 30,
                    right: 30,
                    child: FloatingActionButton(
                      backgroundColor: Colors.black.withValues(alpha: 0.5),
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
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
