import 'dart:typed_data';
import 'package:flutter/material.dart';

class DebugViewer extends StatelessWidget {
  final Uint8List? heatmap;
  final Uint8List? warped;

  const DebugViewer({super.key, required this.heatmap, required this.warped});

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: 40,
      left: 10,
      child: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.5),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (heatmap != null) _buildImage(heatmap!, 'Heatmap'),
            if (warped != null) ...[
              const SizedBox(height: 10),
              _buildImage(warped!, 'Warped Binary'),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildImage(Uint8List bytes, String label) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 4),
        Image.memory(
          bytes,
          width: 128,
          height: 128,
          fit: BoxFit.fill,
          gaplessPlayback: true, // Prevents flickering
        ),
      ],
    );
  }
}
