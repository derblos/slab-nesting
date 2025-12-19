"""
Compatibility wrapper for rectangle-packer to work as rectpack

This allows the existing code to use 'from rectpack import newPacker'
while using the rectangle-packer library under the hood.
"""

from rpack import pack


class Packer:
    """Wrapper class to mimic rectpack.Packer API using rectangle-packer"""

    def __init__(self, mode='online', pack_algo='maxrects', sort_algo='SORT_AREA', rotation=True):
        self.bins = []
        self.rectangles = []
        self.rotation_enabled = rotation

    def add_bin(self, width, height, count=1):
        """Add bins to pack rectangles into"""
        for _ in range(count):
            self.bins.append({'width': width, 'height': height, 'rects': []})

    def add_rect(self, width, height, rid=None):
        """Add a rectangle to be packed"""
        self.rectangles.append({'width': width, 'height': height, 'rid': rid})

    def pack(self):
        """Pack the rectangles into bins"""
        if not self.bins or not self.rectangles:
            return

        # For simplicity, pack into first bin using rpack
        # Extract dimensions
        sizes = [(r['width'], r['height']) for r in self.rectangles]

        # Use rpack to pack
        positions = pack(sizes)

        # Store results in first bin
        for i, (x, y) in enumerate(positions):
            rect_info = self.rectangles[i]
            self.bins[0]['rects'].append({
                'x': x,
                'y': y,
                'width': rect_info['width'],
                'height': rect_info['height'],
                'rid': rect_info['rid']
            })

    def __getitem__(self, index):
        """Get a bin by index"""
        return self.bins[index]


def newPacker(mode='online', pack_algo='maxrects', sort_algo='SORT_AREA', rotation=True):
    """Create a new packer instance (mimics rectpack.newPacker)"""
    return Packer(mode=mode, pack_algo=pack_algo, sort_algo=sort_algo, rotation=rotation)
