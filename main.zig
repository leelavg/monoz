const std = @import("std");
const stdout = std.io.getStdOut().writer();

pub fn main() !void {
    const imageWidth: u16 = 256;
    const imageHeight: u16 = 256;

    try stdout.print("P3\n{d} {d}\n255\n", .{ imageWidth, imageHeight });

    var j: usize = imageHeight - 1;

    while (j > 0) : (j -= 1) {
        var i: usize = 0;
        while (i < imageWidth) : (i += 1) {
            var r = @intToFloat(f64, i) / @intToFloat(f64, imageWidth - 1);
            var g = @intToFloat(f64, j) / @intToFloat(f64, imageHeight - 1);
            const b = 0.25;

            var ir = @floatToInt(u16, 255.999 * r);
            var ig = @floatToInt(u16, 255.999 * g);
            const ib = @floatToInt(u16, 255.999 * b);

            try stdout.print("{d} {d} {d}\n", .{ ir, ig, ib });
        }
    }
}
