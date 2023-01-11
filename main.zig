const std = @import("std");
const math = std.math;
const stdout = std.io.getStdOut().writer();
const print = std.debug.print;
const ArrayList = std.ArrayList;

const infinity = math.inf_f32;
const pi = math.pi;

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

    pub fn at(self: ray, t: f32) point3 {
        return self.orig + expand(t) * self.dir;
    }
};

const hitRecord = struct {
    p: point3,
    n: vec3,
    t: f32,
    frontFace: bool,

    pub fn setFaceNormal(self: *hitRecord, r: ray, outwardNormal: vec3) void {
        self.*.frontFace = getDotPro(r.dir, outwardNormal) < 0;
        self.*.n = if (self.frontFace) outwardNormal else -outwardNormal;
    }
};

const sphere = struct {
    center: point3,
    radius: f32,

    pub fn new(center: point3, radius: f32) sphere {
        return sphere{
            .center = center,
            .radius = radius,
        };
    }

    pub fn hit(self: sphere, r: ray, tMin: f32, tMax: f32) ?hitRecord {
        const oc = r.orig - self.center;
        const a = getDotPro(r.dir, r.dir);
        const half_b = getDotPro(oc, r.dir);
        const c = getDotPro(oc, oc) - self.radius * self.radius;

        const disc = half_b * half_b - a * c;
        if (disc < 0) {
            return null;
        }

        const sqrt = math.sqrt(disc);
        var root = (-half_b - sqrt) / a;
        if (root < tMin or tMax < root) {
            root = (-half_b + sqrt) / a;
            if (root < tMin or tMax < root) {
                return null;
            }
        }

        var rec: hitRecord = undefined;
        rec.t = root;
        rec.p = r.at(rec.t);
        rec.n = (rec.p - self.center) / expand(self.radius);
        const outwardNormal = (rec.p - self.center) / expand(self.radius);
        rec.setFaceNormal(r, outwardNormal);

        return rec;
    }
};

const world = struct {
    spheres: ArrayList(sphere),

    pub fn init(alloc: std.mem.Allocator) world {
        return world{ .spheres = ArrayList(sphere).init(alloc) };
    }

    pub fn deinit(self: *world) void {
        self.spheres.deinit();
    }

    pub fn hit(self: *const world, r: ray, tMin: f32, tMax: f32) ?hitRecord {
        var mayHit: ?hitRecord = null;
        var closestSoFar = tMax;

        for (self.spheres.items) |item| {
            if (item.hit(r, tMin, tMax)) |hitRec| {
                if (hitRec.t < closestSoFar) {
                    mayHit = hitRec;
                    closestSoFar = hitRec.t;
                }
            }
        }

        return mayHit;
    }
};

// get vector from scalar data, basically scalar times a unit vector
fn expand(val: f32) vec {
    return @splat(3, val);
}

fn degToRad(deg: f32) f32 {
    return deg * pi / 180;
}

// return unit vector along a direction
fn getUnitVec(dir: vec) vec {
    return dir / expand(math.sqrt(getDotPro(dir, dir)));
}

// return dot product of two vectors
fn getDotPro(v1: vec, v2: vec) f32 {
    return @reduce(.Add, v1 * v2);
}

// returns background color, a simple gradient
fn rayColor(r: ray, w: world) color {
    if (w.hit(r, 0, infinity)) |rec| {
        return expand(0.5) * (rec.n + color{ 1, 1, 1 });
    }
    const unitDir = getUnitVec(r.dir);
    const t = 0.5 * (unitDir[1] + 1.0);
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

    // World
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    var alloc = arena.allocator();
    var w = world.init(alloc);
    defer w.deinit();

    try w.spheres.append(sphere.new(point3{ 0, 0, -1 }, 0.5));
    try w.spheres.append(sphere.new(point3{ 0, -100.5, -1 }, 100));

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
            const pixelColor: color = rayColor(r, w);
            try writeColor(stdout, pixelColor);
        }
    }
}
