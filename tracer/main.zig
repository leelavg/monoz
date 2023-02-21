const std = @import("std");
const time = std.time;
const fmt = std.fmt;
const mem = std.mem;
const math = std.math;
const bufio = std.io.bufferedWriter;
const stdout = std.io.getStdOut().writer();
const print = std.debug.print;
const arraylist = std.ArrayList;
const randgen = std.rand.DefaultPrng;
const infinity = math.inf_f32;
const pi = math.pi;

const vec = @Vector(3, f32);
const vec3 = vec;
const color = vec;
const point3 = vec;

const ray = struct {
    orig: point3,
    dir: vec3,

    pub fn at(self: ray, t: f32) point3 {
        return self.orig + expand(t) * self.dir;
    }
};

const scatter = struct {
    att: color,
    r: ray,
};

const hitrecord = struct {
    p: point3,
    n: vec3,
    t: f32,
    mp: material,
    frontFace: bool,

    pub fn setFaceNormal(self: *hitrecord, r: ray, outwardNormal: vec3) void {
        self.*.frontFace = getDotPro(r.dir, outwardNormal) < 0;
        self.*.n = if (self.frontFace) outwardNormal else -outwardNormal;
    }
};

const metal = struct {
    albedo: color,
    fuzz: f32,

    fn scat(self: metal, r: ray, rec: hitrecord, rnd: *randgen) ?scatter {
        const reflected = reflect(getUnitVec(r.dir), rec.n);
        const sd = ray{ .orig = rec.p, .dir = reflected + expand(self.fuzz) * getRanUnitSphere(rnd) };
        if (getDotPro(sd.dir, rec.n) > 0) {
            return scatter{
                .att = self.albedo,
                .r = sd,
            };
        }
        return null;
    }
};

const dielectric = struct {
    ir: f32,

    fn reflectance(_: dielectric, cos: f32, refIdx: f32) f32 {
        var r0 = (1 - refIdx) / (1 + refIdx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * math.pow(f32, (1 - cos), 5);
    }
    fn scat(self: dielectric, r: ray, rec: hitrecord, rnd: *randgen) ?scatter {
        const rr = if (rec.frontFace) (1.0 / self.ir) else self.ir;
        const unitDir = getUnitVec(r.dir);
        const cosT = @min(getDotPro(-unitDir, rec.n), 1.0);
        const sinT = math.sqrt(1.0 - cosT * cosT);

        const cannotRef = rr * sinT > 1.0;
        const dir = if (cannotRef or self.reflectance(cosT, rr) > getRanFloat(rnd))
            reflect(unitDir, rec.n)
        else
            refract(unitDir, rec.n, rr);

        return scatter{
            .att = color{ 1.0, 1.0, 1.0 },
            .r = ray{ .orig = rec.p, .dir = dir },
        };
    }
};

const material = union(enum) {
    l: lambertian,
    m: metal,
    d: dielectric,

    pub fn lamber(albedo: color) material {
        return material{ .l = lambertian{ .albedo = albedo } };
    }

    pub fn met(albedo: color, fuzz: f32) material {
        return material{ .m = metal{ .albedo = albedo, .fuzz = if (fuzz < 1) fuzz else 1 } };
    }

    pub fn di(ir: f32) material {
        return material{ .d = dielectric{ .ir = ir } };
    }
};

const lambertian = struct {
    albedo: color,

    pub fn scat(self: lambertian, rec: hitrecord, rnd: *randgen) ?scatter {
        var sd = rec.n + getRanUnitVec(rnd);
        if (isNearZero(sd)) {
            sd = rec.n;
        }
        const s = ray{ .orig = rec.p, .dir = sd };
        const att = self.albedo;
        return scatter{
            .att = att,
            .r = s,
        };
    }
};

const sphere = struct {
    center: point3,
    radius: f32,
    mp: material,

    pub fn new(center: point3, radius: f32, mp: material) sphere {
        return sphere{
            .center = center,
            .radius = radius,
            .mp = mp,
        };
    }

    pub fn hit(self: sphere, r: ray, tMin: f32, tMax: f32) ?hitrecord {
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

        var rec: hitrecord = undefined;
        rec.t = root;
        rec.p = r.at(rec.t);
        rec.n = (rec.p - self.center) / expand(self.radius);
        const outwardNormal = (rec.p - self.center) / expand(self.radius);
        rec.setFaceNormal(r, outwardNormal);
        rec.mp = self.mp;

        return rec;
    }
};

const world = struct {
    spheres: arraylist(sphere),

    pub fn init(alloc: std.mem.Allocator) world {
        return world{ .spheres = arraylist(sphere).init(alloc) };
    }

    pub fn deinit(self: *world) void {
        self.spheres.deinit();
    }

    pub fn hit(self: *const world, r: ray, tMin: f32, tMax: f32) ?hitrecord {
        var mayHit: ?hitrecord = null;
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

const camera = struct {
    origin: point3,
    lowerLeftCor: point3,
    hor: vec3,
    ver: vec3,
    u: vec3,
    v: vec3,
    w: vec3,
    lenRadius: f32,

    const Self = @This();
    pub fn init(
        lookfrom: point3,
        lookat: point3,
        vup: vec3,
        vfov: f32,
        aspectRatio: f32,
        aperture: f32,
        focusDist: f32,
    ) Self {
        const theta = math.degreesToRadians(f32, vfov);
        const h = math.tan(theta / 2);
        const viewPortHeight: f32 = 2.0 * h;
        const viewPortWidth: f32 = aspectRatio * viewPortHeight;

        const w = getUnitVec(lookfrom - lookat);
        const u = getUnitVec(getCrossPro(vup, w));
        const v = getCrossPro(w, u);

        const origin = lookfrom;
        const hor = expand(focusDist) * expand(viewPortWidth) * u;
        const ver = expand(focusDist) * expand(viewPortHeight) * v;
        const half = 2;
        const lowerLeftCor = origin - hor / expand(half) - ver / expand(half) - expand(focusDist) * w;

        return Self{
            .origin = origin,
            .lowerLeftCor = lowerLeftCor,
            .hor = hor,
            .ver = ver,
            .u = u,
            .v = v,
            .w = w,
            .lenRadius = aperture / @as(f32, 2),
        };
    }

    pub fn getRay(self: Self, s: f32, t: f32, rnd: *randgen) ray {
        const rd = expand(self.lenRadius) * getRanUnitDisk(rnd);
        const offset = self.u * expand(rd[0]) + self.v * expand(rd[1]);

        return ray{
            .orig = self.origin + offset,
            .dir = self.lowerLeftCor + expand(s) * self.hor + expand(t) * self.ver - self.origin - offset,
        };
    }
};

fn expand(val: f32) vec {
    return @splat(3, val);
}

fn degToRad(deg: f32) f32 {
    return deg * pi / 180;
}

fn getUnitVec(dir: vec) vec {
    return dir / expand(math.sqrt(getDotPro(dir, dir)));
}

fn getDotPro(v1: vec, v2: vec) f32 {
    return @reduce(.Add, v1 * v2);
}

fn getCrossPro(v1: vec, v2: vec) vec {
    return vec{
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    };
}

fn getRanFloat(rnd: *randgen) f32 {
    return rnd.random().float(f32);
}

fn getRanFloatInRange(rnd: *randgen, min: f32, max: f32) f32 {
    return min + (max - min) * getRanFloat(rnd);
}

fn getRanVec(rnd: *randgen) vec3 {
    return vec3{
        getRanFloat(rnd),
        getRanFloat(rnd),
        getRanFloat(rnd),
    };
}

fn getRanVecInRange(rnd: *randgen, min: f32, max: f32) vec3 {
    return vec3{
        getRanFloatInRange(rnd, min, max),
        getRanFloatInRange(rnd, min, max),
        getRanFloatInRange(rnd, min, max),
    };
}

fn getRanUnitSphere(rnd: *randgen) vec {
    while (true) {
        const p = getRanVecInRange(rnd, -1, 1);
        if (getDotPro(p, p) >= 1) continue;
        return p;
    }
}

fn getRanUnitDisk(rnd: *randgen) vec {
    while (true) {
        const p = vec3{
            getRanFloatInRange(rnd, -1, 1),
            getRanFloatInRange(rnd, -1, 1),
            0,
        };
        if (getDotPro(p, p) >= 1) continue;
        return p;
    }
}

fn getRanUnitVec(rnd: *randgen) vec {
    return getUnitVec(getRanUnitSphere(rnd));
}

fn getRanInHem(n: vec, rnd: *randgen) vec {
    const inUnitSphere = getRanUnitSphere(rnd);
    if (getDotPro(inUnitSphere, n) > 0.0) {
        return inUnitSphere;
    } else {
        return -inUnitSphere;
    }
}

fn getRanScene(alloc: std.mem.Allocator, rnd: *randgen) !world {
    var w: world = world.init(alloc);

    const matGround = material.lamber(color{ 0.5, 0.5, 0.5 });
    try w.spheres.append(sphere.new(point3{ 0, -1000, 0 }, 1000, matGround));

    var a: i8 = -11;
    while (a < 11) : (a += 1) {
        var b: i8 = -11;
        while (b < 11) : (b += 1) {
            const chooseMat = getRanFloat(rnd);
            const center = point3{
                @intToFloat(f32, a) + 0.9 * getRanFloat(rnd),
                0.2,
                @intToFloat(f32, b) + 0.9 * getRanFloat(rnd),
            };

            const dist = center - point3{ 4, 0.2, 0 };
            if (math.sqrt(getDotPro(dist, dist)) > 0.9) {
                const mat: material = if (chooseMat < 0.8) lamber: {
                    const albedo = getRanVec(rnd) * getRanVec(rnd);
                    break :lamber material.lamber(albedo);
                } else if (chooseMat < 0.95) diff: {
                    const albedo = getRanVecInRange(rnd, 0.5, 1);
                    const fuzz = getRanFloatInRange(rnd, 0, 0.5);
                    break :diff material.met(albedo, fuzz);
                } else glass: {
                    break :glass material.di(1.5);
                };
                try w.spheres.append(sphere.new(center, 0.2, mat));
            }
        }
    }

    const mat1 = material.di(1.5);
    try w.spheres.append(sphere.new(point3{ 0, 1, 0 }, 1.0, mat1));

    const mat2 = material.lamber(color{ 0.4, 0.2, 0.1 });
    try w.spheres.append(sphere.new(point3{ -4, 1, 0 }, 1.0, mat2));

    const mat3 = material.met(color{ 0.7, 0.6, 0.5 }, 0.0);
    try w.spheres.append(sphere.new(point3{ 4, 1, 0 }, 1.0, mat3));

    return w;
}

fn isNearZero(v: vec) bool {
    const s = 1e-8;
    return math.fabs(v[0]) < s and math.fabs(v[1]) < s and math.fabs(v[2]) < s;
}

fn reflect(v: vec, n: vec) vec {
    return v - expand(2 * getDotPro(v, n)) * n;
}

fn refract(uv: vec, n: vec, eoe: f32) vec {
    const cosT = @min(getDotPro(-uv, n), 1.0);
    const rPer = expand(eoe) * (uv + expand(cosT) * n);
    const rPar = expand(-math.sqrt(math.fabs(1.0 - getDotPro(rPer, rPer)))) * n;
    return rPer + rPar;
}

fn rayColor(r: ray, w: world, rnd: *randgen, depth: u8) color {
    if (depth <= 0) return color{ 0, 0, 0 };

    if (w.hit(r, 0.001, infinity)) |rec| {
        const s = switch (rec.mp) {
            material.l => |l| l.scat(rec, rnd),
            material.m => |m| m.scat(r, rec, rnd),
            material.d => |d| d.scat(r, rec, rnd),
        };
        if (s) |scat| return scat.att * rayColor(scat.r, w, rnd, depth - 1);
        return color{ 0, 0, 0 };
    }
    const unitDir = getUnitVec(r.dir);
    const t = 0.5 * (unitDir[1] + 1.0);
    return expand(1 - t) * color{ 1.0, 1.0, 1.0 } + expand(t) * color{ 0.5, 0.7, 1.0 };
}

fn clamp(x: f32, min: f32, max: f32) f32 {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

fn writeColor(writer: anytype, pixelColor: color, samples: u16) !void {
    var r = pixelColor[0];
    var g = pixelColor[1];
    var b = pixelColor[2];

    const scale = 1.0 / @intToFloat(f32, samples);
    r = math.sqrt(scale * r);
    g = math.sqrt(scale * g);
    b = math.sqrt(scale * b);

    const ir = @floatToInt(u8, 255.999 * clamp(r, 0.0, 0.999));
    const ig = @floatToInt(u8, 255.999 * clamp(g, 0.0, 0.999));
    const ib = @floatToInt(u8, 255.999 * clamp(b, 0.0, 0.999));
    try writer.print("{d} {d} {d}\n", .{ ir, ig, ib });
}

pub fn main() !void {

    // Allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();
    var alloc = arena.allocator();

    // Image
    var file: []const u8 = "/tmp/image.ppm";
    var imageHeight: u16 = 675;
    var imageWidth: u16 = 1200;
    var samples: u16 = 500;
    var maxDepth: u8 = 50;

    // Args
    var args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    args = args[1..args.len];
    for (args) |arg| {
        var split = mem.split(u8, arg, "=");
        var key = mem.trimLeft(u8, split.first(), "-");
        var val = split.rest();

        if (mem.eql(u8, key, "height")) {
            imageHeight = try std.fmt.parseUnsigned(u16, val, 10);
        } else if (mem.eql(u8, key, "width")) {
            imageWidth = try std.fmt.parseUnsigned(u16, val, 10);
        } else if (mem.eql(u8, key, "samples")) {
            samples = try std.fmt.parseUnsigned(u16, val, 10);
        } else if (mem.eql(u8, key, "depth")) {
            maxDepth = try std.fmt.parseUnsigned(u8, val, 10);
        } else if (mem.eql(u8, key, "file")) {
            file = val;
        }
    }

    const aspectRatio: f32 = @intToFloat(f32, imageWidth) / @intToFloat(f32, imageHeight);

    var rnd = std.rand.DefaultPrng.init(0);

    // World
    var w = try getRanScene(alloc, &rnd);
    defer w.deinit();

    // Camera
    const lookfrom = point3{ 13, 2, 3 };
    const lookat = point3{ 0, 0, 0 };
    const vup = vec3{ 0, 1, 0 };
    const distToFocus = 10.0;
    const vfov = 20;
    const aperture = 0.1;
    const cam = camera.init(
        lookfrom,
        lookat,
        vup,
        vfov,
        aspectRatio,
        aperture,
        distToFocus,
    );

    // Render
    var j: u16 = imageHeight - 1;
    var out_file = try std.fs.cwd().createFile(file, .{ .read = false });
    defer out_file.close();

    var bufWriter = bufio(out_file.writer());
    var out = bufWriter.writer();
    try out.print("P3\n{d} {d}\n255\n", .{ imageWidth, imageHeight });

    const clr = "\x1B[K";
    var timer = try time.Timer.start();
    while (j > 0) : (j -= 1) {
        var i: u16 = 0;
        print("\rScanning remaining lines: {d}{s}", .{ j, clr });
        while (i < imageWidth) : (i += 1) {
            var pixelColor: color = color{ 0, 0, 0 };
            var s: u16 = 0;
            while (s <= samples) : (s += 1) {
                const u = (@intToFloat(f32, i) + getRanFloat(&rnd)) / @intToFloat(f32, imageWidth - 1);
                const v = (@intToFloat(f32, j) + getRanFloat(&rnd)) / @intToFloat(f32, imageHeight - 1);

                // the ray is originating & passing through origin in the direction with two vectors imposed on the screen
                const r: ray = cam.getRay(u, v, &rnd);
                pixelColor += rayColor(r, w, &rnd, maxDepth);
            }
            try writeColor(out, pixelColor, samples);
        }
    }
    try bufWriter.flush();
    print("\r=== Done rendering (took: {s}) ==={s}\n", .{ fmt.fmtDuration(timer.read()), clr });
}
