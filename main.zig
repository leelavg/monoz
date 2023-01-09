const std = @import("std");
const stdout = std.io.getStdOut().writer();
const color = @Vector(3, f64);

fn writeColor(writer: anytype, pixelColor: color) !void {
    const ir = @floatToInt(u16, 255.999 * pixelColor[0]);
    const ig = @floatToInt(u16, 255.999 * pixelColor[1]);
    const ib = @floatToInt(u16, 255.999 * pixelColor[2]);
    try writer.print("{d} {d} {d}\n", .{ ir, ig, ib });
}

pub fn main() !void {
    const imageWidth: u16 = 256;
    const imageHeight: u16 = 256;

    try stdout.print("P3\n{d} {d}\n255\n", .{ imageWidth, imageHeight });

    var j: u8 = imageHeight - 1;

    while (j > 0) : (j -= 1) {
        var i: u16 = 0;
        while (i < imageWidth) : (i += 1) {
            const r = @intToFloat(f64, i) / @intToFloat(f64, imageWidth - 1);
            const g = @intToFloat(f64, j) / @intToFloat(f64, imageHeight - 1);
            const b = 0.25;

            const pixelColor: color = color{ r, g, b };
            try writeColor(stdout, pixelColor);
        }
    }
}
