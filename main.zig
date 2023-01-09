const std = @import("std");
const math = std.math;
const stdout = std.io.getStdOut().writer();
const print = std.debug.print;

// base type for all vector operations and identifiers
// return type for all functions operating on vector
const vec = @Vector(3, f32);

// any adhoc vector type
const vec3 = vec;

// RGB color representation
const color = vec;

// a point in 3D space
const point3 = vec;

// ray which has a origin point and a direction in 3D space
const ray = struct {
    // P(t) = A + tb
    // P = 3D position along a line, t = ray parameter, A = ray origin, b = ray direction
    orig: point3,
    dir: vec3,

    const Self = @This();
    pub fn at(self: Self, t: f32) point3 {
        return self.orig + expand(t) * self.dir;
    }
};

// get vector from scalar data, basically scalar times a unit vector
fn expand(val: f32) vec {
    return @splat(3, val);
}

// return unit vector along a direction
fn getUnitVec(dir: vec) vec {
    return dir / expand(math.sqrt(getDotPro(dir, dir)));
}

// return dot product of two vectors
fn getDotPro(v1: vec, v2: vec) f32 {
    return @reduce(.Add, v1 * v2);
}

fn hitSphere(center: point3, radius: f32, r: ray) f32 {
    // t^2b.b + 2tb.(A-C) + (A-C).(A-C) - r^2 = 0
    // solve for t
    const oc = r.orig - center;
    const a = getDotPro(r.dir, r.dir);
    const b = 2.0 * getDotPro(oc, r.dir);
    const c = getDotPro(oc, oc) - radius * radius;
    const disc = b * b - 4 * a * c;
    // positive = 2 roots, exact zero = 1 root, negative = 0 roots
    // roots == number of places that ray hits the sphere
    if (disc < 0) return -1 else return (-b - math.sqrt(disc) / (2 * a));
}

// returns background color, a simple gradient
fn rayColor(r: ray) color {
    var t = hitSphere(point3{ 0, 0, -1 }, 0.5, r);
    if (t > 0) {
        const N = getUnitVec(r.at(t) - vec3{ 0, 0, -1 });
        return expand(0.5) * color{ N[0] + 1, N[1] + 1, N[2] + 1 };
    }
    const unitDir = getUnitVec(r.dir);
    t = 0.5 * (unitDir[1] + 1.0);
    // blendedvalue = (1-t)*startVal + t*endVal
    return expand(1 - t) * color{ 1.0, 1.0, 1.0 } + expand(t) * color{ 0.5, 0.7, 1.0 };
}

// output the rgb info to writer, calculations correspond to how much percent of
// a color from rgb represents a single pixel
fn writeColor(writer: anytype, pixelColor: color) !void {
    const ir = @floatToInt(u16, 255.999 * pixelColor[0]);
    const ig = @floatToInt(u16, 255.999 * pixelColor[1]);
    const ib = @floatToInt(u16, 255.999 * pixelColor[2]);
    try writer.print("{d} {d} {d}\n", .{ ir, ig, ib });
}

pub fn main() !void {

    // Image
    const aspectRatio: f32 = 16.0 / 9.0;
    const imageWidth: u16 = 400;
    const imageHeight: u16 = @floatToInt(u16, @intToFloat(f32, imageWidth) / aspectRatio);

    // Camera
    const viewPortHeight: f32 = 2.0;
    const viewPortWidth: f32 = aspectRatio * viewPortHeight;

    // distance b/n projection plane and the projection point
    const focalLen: f32 = 1.0;

    // camera/eye origin
    const origin: point3 = point3{ 0, 0, 0 };
    const hor: vec3 = vec3{ viewPortWidth, 0, 0 };
    const ver: vec3 = vec3{ 0, viewPortHeight, 0 };
    const half: f32 = 2;
    const lowerLeftCor: vec3 = origin - hor / expand(half) - ver / expand(half) - vec3{ 0, 0, focalLen };

    // Render
    try stdout.print("P3\n{d} {d}\n255\n", .{ imageWidth, imageHeight });
    var j: u8 = imageHeight - 1;

    while (j > 0) : (j -= 1) {
        var i: u16 = 0;
        while (i < imageWidth) : (i += 1) {
            const u = @intToFloat(f32, i) / @intToFloat(f32, imageWidth - 1);
            const v = @intToFloat(f32, j) / @intToFloat(f32, imageHeight - 1);

            // the ray is originating & passing through origin in the direction with two vectors imposed on the screen
            const r: ray = ray{ .orig = origin, .dir = lowerLeftCor + expand(u) * hor + expand(v) * ver - origin };
            const pixelColor: color = rayColor(r);
            try writeColor(stdout, pixelColor);
        }
    }
}
