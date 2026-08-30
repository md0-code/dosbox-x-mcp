/* DOSBox-X MCP companion protocol v2.
 *
 * Included by sdlmain.cpp after the v1 keyboard hook. The channel is disabled
 * unless DOSBOX_X_MCP_TOKEN is present. Windows uses a local named pipe;
 * POSIX uses a same-uid Unix socket in a mode-0700 runtime directory. Requests
 * and responses are length-prefixed UTF-8 JSON. All emulator mutation occurs
 * from DOSBoxMCPV2Pump on the SDL/emulation thread.
 */
#include <algorithm>
#include <stdint.h>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "vga.h"

#if !defined(WIN32)
# include <errno.h>
# include <poll.h>
# include <sys/socket.h>
# include <sys/un.h>
#endif

struct DOSBoxMCPV2TimelineEvent {
    uint64_t at;
    unsigned int order;
    std::string kind;
    int key;
    std::vector<int> keys;
    uint64_t actual;
    uint64_t pic_us;
    uint64_t video_frame;
    uint64_t cpu_cycles;
    uint64_t irq_before;
    uint32_t physical;
    uint32_t length;
    uint32_t capture_bytes;
    uint32_t hit_limit;
    std::string acknowledgement;
    std::string observation;
};

static std::string dosbox_mcp_v2_token;
static std::string dosbox_mcp_v2_session;
static std::string dosbox_mcp_v2_input;
static uint64_t dosbox_mcp_v2_frame = 0;
static uint64_t dosbox_mcp_v2_sequence = 0;
static bool dosbox_mcp_v2_active = false;
static std::string dosbox_mcp_v2_request_id;
static std::string dosbox_mcp_v2_clock;
static uint64_t dosbox_mcp_v2_start = 0;
static uint64_t dosbox_mcp_v2_expire_after = 0;
static uint64_t dosbox_mcp_v2_expire_pic_us = 0;
static std::vector<DOSBoxMCPV2TimelineEvent> dosbox_mcp_v2_timeline;
static size_t dosbox_mcp_v2_next = 0;
static bool dosbox_mcp_v2_finish_wait = false;
static uint64_t dosbox_mcp_v2_finish_frame = 0;
static uint64_t dosbox_mcp_v2_irq1_sequence = 0;
static std::set<int> dosbox_mcp_v2_pressed;
static bool dosbox_mcp_v2_restore_wait = false;
static uint64_t dosbox_mcp_v2_restore_frame = 0;
static uint64_t dosbox_mcp_v2_restore_pic = 0;
static uint64_t dosbox_mcp_v2_restore_cycles = 0;
static bool dosbox_mcp_v2_video_wait = false;
static uint64_t dosbox_mcp_v2_video_target = 0;
static size_t dosbox_mcp_v2_video_limit = 0;

extern void SaveGameState_Run(void);
extern void LoadGameState_Run(void);

void DOSBoxMCPKeyboardIRQ1Observed(void) { ++dosbox_mcp_v2_irq1_sequence; }

struct DOSBoxMCPV2WriteEvent {
    uint32_t physical;
    uint32_t before;
    uint32_t after;
    unsigned int width;
    uint16_t cs, ip, ax, bx, cx, dx, si, di, bp, sp, ds, es, ss;
    uint64_t counter, pic_us, video_frame, cpu_cycles;
};

bool dosbox_mcp_write_watch_enabled = false;
static bool dosbox_mcp_v2_watch_in_timeline = false;
static uint32_t dosbox_mcp_v2_watch_start = 0;
static uint32_t dosbox_mcp_v2_watch_end = 0;
static uint64_t dosbox_mcp_v2_watch_until = 0;
static size_t dosbox_mcp_v2_watch_limit = 0;
static std::string dosbox_mcp_v2_watch_clock;
static std::vector<DOSBoxMCPV2WriteEvent> dosbox_mcp_v2_writes;
static bool dosbox_mcp_v2_writes_truncated = false;

#if defined(WIN32)
static HANDLE dosbox_mcp_v2_pipe = INVALID_HANDLE_VALUE;
static bool dosbox_mcp_v2_connected = false;
#else
static int dosbox_mcp_v2_server = -1;
static int dosbox_mcp_v2_client = -1;
static std::string dosbox_mcp_v2_socket_path;
#endif

static std::string DOSBoxMCPV2Escape(const std::string &text) {
    std::string out;
    for (std::string::const_iterator it = text.begin(); it != text.end(); ++it) {
        const unsigned char c = (unsigned char)*it;
        if (c == '\\' || c == '"') { out += '\\'; out += (char)c; }
        else if (c >= 0x20) out += (char)c;
    }
    return out;
}

static std::string DOSBoxMCPV2Hex(const uint8_t *data, size_t size) {
    static const char digits[] = "0123456789abcdef";
    std::string result(size * 2, '0');
    for (size_t i = 0; i < size; ++i) {
        result[i * 2] = digits[data[i] >> 4];
        result[i * 2 + 1] = digits[data[i] & 15];
    }
    return result;
}

static std::string DOSBoxMCPV2VideoObservation(size_t requested_bytes) {
    const size_t limit = std::min<size_t>(requested_bytes, 1024 * 1024);
    const size_t raw_size = vga.mem.linear == NULL ? 0 : std::min<size_t>(vga.mem.memsize, limit);
    std::vector<uint8_t> palette(768);
    for (size_t i = 0; i < 256; ++i) {
        palette[i * 3] = vga.dac.rgb[i].red;
        palette[i * 3 + 1] = vga.dac.rgb[i].green;
        palette[i * 3 + 2] = vga.dac.rgb[i].blue;
    }
    const char *layout = "unsupported";
    if (vga.mode == M_VGA && vga.config.chained) layout = "dosbox_chain4_sparse";
    else if (vga.mode == M_VGA || vga.mode == M_EGA || vga.mode == M_LIN4 || vga.mode == M_PACKED4) layout = "plane_interleaved";
    else if (vga.mode == M_TEXT) layout = "text_plane_interleaved";
    else if (vga.mode >= M_LIN8 && vga.mode <= M_LIN32) layout = "linear_packed";
    const bool packed_linear = vga.mode >= M_LIN8 && vga.mode <= M_LIN32;
    const unsigned int logical_height = (unsigned int)(!packed_linear && vga.draw.doublescan_effect && vga.draw.height > 1 ? vga.draw.height / 2 : vga.draw.height);
    unsigned int guest_bpp = 0;
    if (vga.mode == M_VGA || vga.mode == M_LIN8) guest_bpp = 8;
    else if (vga.mode == M_EGA || vga.mode == M_LIN4 || vga.mode == M_PACKED4) guest_bpp = 4;
    else if (vga.mode == M_LIN15) guest_bpp = 15;
    else if (vga.mode == M_LIN16) guest_bpp = 16;
    else if (vga.mode == M_LIN24) guest_bpp = 24;
    else if (vga.mode == M_LIN32) guest_bpp = 32;
    const unsigned int plane_pitch = (unsigned int)(vga.config.scan_len * 2);
    const unsigned int guest_pitch = vga.mode == M_VGA && vga.config.chained ? plane_pitch * 4 :
        (packed_linear ? (unsigned int)vga.draw.address_add : plane_pitch);
    std::ostringstream out;
    out << "{\"mode_id\":" << (unsigned int)vga.mode << ",\"mode_name\":\"" << DOSBoxMCPV2Escape(mode_texts[vga.mode])
        << "\",\"width\":" << vga.draw.width << ",\"height\":" << logical_height
        << ",\"render_height\":" << vga.draw.height << ",\"render_bpp\":" << vga.draw.bpp
        << ",\"doublescan\":" << (vga.draw.doublescan_effect ? "true" : "false")
        << ",\"pitch\":" << guest_pitch << ",\"plane_pitch\":" << plane_pitch
        << ",\"crtc_scan_len\":" << vga.config.scan_len << ",\"start_address\":" << vga.config.real_start
        << ",\"bpp\":" << guest_bpp << ",\"chained\":" << (vga.config.chained ? "true" : "false")
        << ",\"text_columns\":" << (vga.mode == M_TEXT ? (unsigned int)vga.draw.blocks : 0)
        << ",\"text_rows\":" << (vga.mode == M_TEXT && vga.draw.address_line_total ? (unsigned int)vga.draw.height / (unsigned int)vga.draw.address_line_total : 0)
        << ",\"font_height\":" << (vga.mode == M_TEXT ? (unsigned int)vga.draw.address_line_total : 0)
        << ",\"text_cell_stride\":" << (vga.mode == M_TEXT ? 8 : 0)
        << ",\"char9dot\":" << (vga.mode == M_TEXT && vga.draw.char9dot ? "true" : "false")
        << ",\"layout\":\"" << layout << "\",\"memory_bytes\":" << vga.mem.memsize
        << ",\"raw_bytes\":" << raw_size << ",\"raw_hex\":\"";
    if (raw_size) out << DOSBoxMCPV2Hex(vga.mem.linear, raw_size);
    out << "\",\"dac_bits\":" << (unsigned int)vga.dac.bits << ",\"dac_hex\":\""
        << DOSBoxMCPV2Hex(&palette[0], palette.size()) << "\"}";
    return out.str();
}

static std::string DOSBoxMCPV2MemoryObservation(uint32_t physical, uint32_t length) {
    if (MemBase == NULL || length == 0 || physical > MemSize || length > MemSize - physical)
        return "{\"error\":\"memory range unavailable\"}";
    std::ostringstream out;
    out << "{\"physical_start\":" << physical << ",\"bytes\":" << length << ",\"hex\":\""
        << DOSBoxMCPV2Hex(MemBase + physical, length) << "\"}";
    return out.str();
}

static std::string DOSBoxMCPV2String(const std::string &json, const char *key) {
    const std::string needle = std::string("\"") + key + "\"";
    size_t at = json.find(needle);
    if (at == std::string::npos) return std::string();
    at = json.find(':', at + needle.size());
    if (at == std::string::npos) return std::string();
    at = json.find('"', at + 1);
    if (at == std::string::npos) return std::string();
    std::string out;
    for (++at; at < json.size(); ++at) {
        if (json[at] == '"') break;
        if (json[at] == '\\' && at + 1 < json.size()) ++at;
        out += json[at];
    }
    return out;
}

static uint64_t DOSBoxMCPV2Integer(const std::string &json, const char *key, uint64_t fallback = 0) {
    const std::string needle = std::string("\"") + key + "\"";
    size_t at = json.find(needle);
    if (at == std::string::npos) return fallback;
    at = json.find(':', at + needle.size());
    if (at == std::string::npos) return fallback;
    while (++at < json.size() && (json[at] == ' ' || json[at] == '\t')) {}
    uint64_t value = 0;
    bool any = false;
    while (at < json.size() && json[at] >= '0' && json[at] <= '9') {
        any = true;
        value = value * 10 + (uint64_t)(json[at++] - '0');
    }
    return any ? value : fallback;
}

static std::vector<int> DOSBoxMCPV2Integers(const std::string &json, const char *key) {
    std::vector<int> result;
    const std::string needle = std::string("\"") + key + "\"";
    size_t at = json.find(needle);
    if (at == std::string::npos || (at = json.find('[', at)) == std::string::npos) return result;
    const size_t end = json.find(']', at);
    if (end == std::string::npos) return result;
    while (at < end) {
        while (++at < end && !(json[at] >= '0' && json[at] <= '9')) {}
        if (at >= end) break;
        int value = 0;
        while (at < end && json[at] >= '0' && json[at] <= '9') value = value * 10 + (json[at++] - '0');
        result.push_back(value);
        --at;
    }
    return result;
}

static uint64_t DOSBoxMCPV2PICUS(void) {
    const double value = PIC_FullIndex() * 1000.0;
    return value > 0.0 ? (uint64_t)value : 0;
}

static uint64_t DOSBoxMCPV2Cycles(void) {
    /* A monotonic emulator-cycle coordinate: completed PIC milliseconds plus
     * the cycle position inside the current millisecond. */
    const uint64_t raw = (uint64_t)PIC_Ticks * (uint64_t)CPU_CycleMax + (uint64_t)PIC_TickIndexND();
    static bool initialized = false;
    static uint64_t previous = 0;
    static uint64_t accumulated = 0;
    if (!initialized) {
        previous = raw;
        initialized = true;
        return 0;
    }
    if (raw >= previous) accumulated += raw - previous;
    previous = raw;
    return accumulated;
}

static uint64_t DOSBoxMCPV2Counter(const std::string &clock) {
    if (clock == "video_frame") return dosbox_mcp_v2_frame;
    if (clock == "cpu_cycles") return DOSBoxMCPV2Cycles();
    if (clock == "wall_clock") return (uint64_t)SDL_GetTicks() * 1000ULL;
    return DOSBoxMCPV2PICUS();
}

void DOSBoxMCPObserveHostWrite(HostPt off, uint32_t value, unsigned int width) {
    if (!dosbox_mcp_write_watch_enabled || MemBase == NULL) return;
    const uintptr_t address = (uintptr_t)off;
    const uintptr_t base = (uintptr_t)MemBase;
    if (address < base || address >= base + MemSize) return;
    const uint32_t physical = (uint32_t)(address - base);
    if (physical < dosbox_mcp_v2_watch_start || physical >= dosbox_mcp_v2_watch_end) return;
    if (dosbox_mcp_v2_writes.size() >= dosbox_mcp_v2_watch_limit) {
        dosbox_mcp_v2_writes_truncated = true;
        return;
    }
    DOSBoxMCPV2WriteEvent event;
    event.physical = physical;
    event.width = width;
    event.before = width == 1 ? host_readb(off) : (width == 2 ? host_readw(off) : host_readd(off));
    event.after = value;
    event.cs = SegValue(cs); event.ip = reg_ip;
    event.ax = reg_ax; event.bx = reg_bx; event.cx = reg_cx; event.dx = reg_dx;
    event.si = reg_si; event.di = reg_di; event.bp = reg_bp; event.sp = reg_sp;
    event.ds = SegValue(ds); event.es = SegValue(es); event.ss = SegValue(ss);
    event.counter = DOSBoxMCPV2Counter(dosbox_mcp_v2_watch_clock);
    event.pic_us = DOSBoxMCPV2PICUS(); event.video_frame = dosbox_mcp_v2_frame; event.cpu_cycles = DOSBoxMCPV2Cycles();
    dosbox_mcp_v2_writes.push_back(event);
}

static std::string DOSBoxMCPV2Counters(void) {
    std::ostringstream out;
    out << "{\"pic_us\":" << DOSBoxMCPV2PICUS()
        << ",\"video_frame\":" << dosbox_mcp_v2_frame
        << ",\"cpu_cycles\":" << DOSBoxMCPV2Cycles() << "}";
    return out.str();
}

static void DOSBoxMCPV2ReleaseAll(void) {
    for (std::set<int>::reverse_iterator it = dosbox_mcp_v2_pressed.rbegin(); it != dosbox_mcp_v2_pressed.rend(); ++it)
        if (*it > KBD_NONE && *it < KBD_LAST) KEYBOARD_AddKey((KBD_KEYS)*it, false);
    dosbox_mcp_v2_pressed.clear();
}

static void DOSBoxMCPV2CloseClient(void) {
#if defined(WIN32)
    if (dosbox_mcp_v2_pipe != INVALID_HANDLE_VALUE && dosbox_mcp_v2_connected) {
        FlushFileBuffers(dosbox_mcp_v2_pipe);
        DisconnectNamedPipe(dosbox_mcp_v2_pipe);
    }
    dosbox_mcp_v2_connected = false;
#else
    if (dosbox_mcp_v2_client >= 0) close(dosbox_mcp_v2_client);
    dosbox_mcp_v2_client = -1;
#endif
    dosbox_mcp_v2_input.clear();
}

#if !defined(WIN32)
static bool DOSBoxMCPV2WouldBlock() {
    if (errno == EAGAIN) return true;
# if EWOULDBLOCK != EAGAIN
    if (errno == EWOULDBLOCK) return true;
# endif
    return false;
}
#endif

static bool DOSBoxMCPV2ClientAlive(void) {
#if defined(WIN32)
    if (!dosbox_mcp_v2_connected) return true;
    DWORD available = 0;
    if (PeekNamedPipe(dosbox_mcp_v2_pipe, NULL, 0, NULL, &available, NULL)) return true;
    DOSBoxMCPV2ReleaseAll();
    dosbox_mcp_v2_active = false;
    dosbox_mcp_v2_finish_wait = false;
    dosbox_mcp_v2_restore_wait = false;
    dosbox_mcp_write_watch_enabled = false;
    dosbox_mcp_v2_watch_in_timeline = false;
    dosbox_mcp_v2_video_wait = false;
    dosbox_mcp_v2_connected = false;
    DisconnectNamedPipe(dosbox_mcp_v2_pipe);
    dosbox_mcp_v2_input.clear();
    return false;
#else
    if (dosbox_mcp_v2_client < 0) return true;
    char byte = 0;
    const ssize_t result = recv(dosbox_mcp_v2_client, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
    if (result > 0 || (result < 0 && DOSBoxMCPV2WouldBlock())) return true;
    DOSBoxMCPV2ReleaseAll();
    dosbox_mcp_v2_active = false;
    dosbox_mcp_v2_finish_wait = false;
    dosbox_mcp_v2_restore_wait = false;
    dosbox_mcp_write_watch_enabled = false;
    dosbox_mcp_v2_watch_in_timeline = false;
    dosbox_mcp_v2_video_wait = false;
    DOSBoxMCPV2CloseClient();
    return false;
#endif
}

#if defined(WIN32)
static bool DOSBoxMCPV2WriteAll(const void *data, size_t size) {
    const uint8_t *next = (const uint8_t *)data;
    while (size) {
        DWORD done = 0;
        const DWORD chunk = size > 0xffffffffu ? 0xffffffffu : (DWORD)size;
        if (!WriteFile(dosbox_mcp_v2_pipe, next, chunk, &done, NULL) || done == 0) return false;
        next += done;
        size -= done;
    }
    return true;
}
#else
static bool DOSBoxMCPV2WriteAll(const void *data, size_t size) {
    const uint8_t *next = (const uint8_t *)data;
    while (size) {
# if defined(MSG_NOSIGNAL)
        const ssize_t done = send(dosbox_mcp_v2_client, next, size, MSG_NOSIGNAL);
# else
        const ssize_t done = send(dosbox_mcp_v2_client, next, size, 0);
# endif
        if (done < 0 && errno == EINTR) continue;
        if (done < 0 && DOSBoxMCPV2WouldBlock()) {
            struct pollfd ready;
            ready.fd = dosbox_mcp_v2_client;
            ready.events = POLLOUT;
            ready.revents = 0;
            int polled;
            do { polled = poll(&ready, 1, 5000); } while (polled < 0 && errno == EINTR);
            if (polled > 0 && (ready.revents & POLLOUT)) continue;
        }
        if (done <= 0) return false;
        next += done;
        size -= (size_t)done;
    }
    return true;
}
#endif

static void DOSBoxMCPV2Write(const std::string &body) {
    const uint32_t size = (uint32_t)body.size();
    unsigned char header[4] = {(unsigned char)size, (unsigned char)(size >> 8), (unsigned char)(size >> 16), (unsigned char)(size >> 24)};
    if (!DOSBoxMCPV2WriteAll(header, 4) || !DOSBoxMCPV2WriteAll(body.data(), size))
        DOSBoxMCPV2ReleaseAll();
    DOSBoxMCPV2CloseClient();
}

static std::string DOSBoxMCPV2Envelope(const std::string &request, const std::string &result, const std::string &events = "[]") {
    std::ostringstream out;
    out << "{\"protocol\":2,\"status\":\"ok\",\"request_id\":\"" << DOSBoxMCPV2Escape(request)
        << "\",\"session_id\":\"" << dosbox_mcp_v2_session << "\",\"result\":" << result
        << ",\"events\":" << events << "}";
    return out.str();
}

static void DOSBoxMCPV2Error(const std::string &request, const std::string &message) {
    std::ostringstream out;
    out << "{\"protocol\":2,\"status\":\"error\",\"request_id\":\"" << DOSBoxMCPV2Escape(request)
        << "\",\"session_id\":\"" << dosbox_mcp_v2_session << "\",\"error\":\"" << DOSBoxMCPV2Escape(message) << "\",\"events\":[]}";
    DOSBoxMCPV2Write(out.str());
}

static std::string DOSBoxMCPV2Event(const char *type, uint64_t pic, uint64_t frame, uint64_t cycles) {
    std::ostringstream out;
    out << "{\"session_id\":\"" << dosbox_mcp_v2_session << "\",\"sequence\":"
        << ++dosbox_mcp_v2_sequence << ",\"type\":\"" << type
        << "\",\"counters\":{\"pic_us\":" << pic << ",\"video_frame\":" << frame
        << ",\"cpu_cycles\":" << cycles << "}}";
    return out.str();
}

static bool DOSBoxMCPV2ParseTimeline(const std::string &json) {
    dosbox_mcp_v2_timeline.clear();
    uint64_t captured_bytes = 0;
    size_t at = json.find("\"events\"");
    if (at == std::string::npos || (at = json.find('[', at)) == std::string::npos) return false;
    int array_depth = 1;
    for (++at; at < json.size() && array_depth; ++at) {
        if (json[at] == ']') { --array_depth; continue; }
        if (json[at] != '{') continue;
        const size_t start = at;
        int depth = 1;
        bool quoted = false;
        for (++at; at < json.size() && depth; ++at) {
            if (json[at] == '"' && (at == 0 || json[at - 1] != '\\')) quoted = !quoted;
            if (!quoted && json[at] == '{') ++depth;
            else if (!quoted && json[at] == '}') --depth;
        }
        if (depth) return false;
        const std::string object = json.substr(start, at - start);
        DOSBoxMCPV2TimelineEvent event;
        event.at = DOSBoxMCPV2Integer(object, "at");
        event.order = (unsigned int)DOSBoxMCPV2Integer(object, "order", (uint64_t)dosbox_mcp_v2_timeline.size());
        event.kind = DOSBoxMCPV2String(object, "kind");
        event.key = (int)DOSBoxMCPV2Integer(object, "key_id");
        event.keys = DOSBoxMCPV2Integers(object, "key_ids");
        event.actual = 0;
        event.pic_us = event.video_frame = event.cpu_cycles = 0;
        event.irq_before = 0;
        event.physical = (uint32_t)DOSBoxMCPV2Integer(object, "physical_start");
        event.length = (uint32_t)DOSBoxMCPV2Integer(object, "length");
        event.capture_bytes = (uint32_t)DOSBoxMCPV2Integer(object, "capture_bytes");
        event.hit_limit = (uint32_t)DOSBoxMCPV2Integer(object, "hit_limit");
        captured_bytes += event.capture_bytes;
        if (event.kind == "memory_snapshot")
            captured_bytes += event.length;
        if (event.kind == "write_watch_start") {
            if (event.hit_limit == 0 || event.hit_limit > 1000) return false;
            captured_bytes += (uint64_t)event.hit_limit * 256;
        }
        if (event.kind.empty()) return false;
        dosbox_mcp_v2_timeline.push_back(event);
        --at;
    }
    std::stable_sort(dosbox_mcp_v2_timeline.begin(), dosbox_mcp_v2_timeline.end(),
        [](const DOSBoxMCPV2TimelineEvent &a, const DOSBoxMCPV2TimelineEvent &b) {
            return a.at < b.at || (a.at == b.at && a.order < b.order);
        });
    return !dosbox_mcp_v2_timeline.empty() && dosbox_mcp_v2_timeline.size() <= 4096 && captured_bytes <= 1024 * 1024;
}

static void DOSBoxMCPV2Begin(const std::string &json) {
    const std::string request = DOSBoxMCPV2String(json, "request_id");
    if (DOSBoxMCPV2Integer(json, "protocol") != 2) { DOSBoxMCPV2Error(request, "unsupported protocol; expected 2"); return; }
    if (DOSBoxMCPV2String(json, "token") != dosbox_mcp_v2_token) { DOSBoxMCPV2Error(request, "authentication failed"); return; }
    const std::string operation = DOSBoxMCPV2String(json, "operation");
    if (operation == "negotiate" || operation == "status") {
        std::ostringstream result;
        result << "{\"build\":{\"dosbox_x_version\":\"" << VERSION << "\",\"source_tag\":\"dosbox-x-v2026.07.02\",\"companion_protocol\":2},"
               "\"capabilities\":{\"timeline\":true,\"authentic_keyboard\":true,\"save_load_ack\":true,"
               "\"write_watch\":true,\"video_snapshot\":true,\"transport\":\""
#if defined(WIN32)
               << "windows_named_pipe"
#else
               << "unix_domain_socket"
#endif
               << "\"},\"counters\":" << DOSBoxMCPV2Counters() << "}";
        DOSBoxMCPV2Write(DOSBoxMCPV2Envelope(request, result.str()));
        return;
    }
    if (operation == "save_state") {
        const uint64_t start_pic = DOSBoxMCPV2PICUS(), start_frame = dosbox_mcp_v2_frame, start_cycles = DOSBoxMCPV2Cycles();
        SaveGameState_Run();
        const std::string started_event = DOSBoxMCPV2Event("save_started", start_pic, start_frame, start_cycles);
        const std::string completed_event = DOSBoxMCPV2Event("save_completed", DOSBoxMCPV2PICUS(), dosbox_mcp_v2_frame, DOSBoxMCPV2Cycles());
        std::ostringstream events;
        events << '[' << started_event << ',' << completed_event << ']';
        DOSBoxMCPV2Write(DOSBoxMCPV2Envelope(request, "{\"saved\":true}", events.str()));
        return;
    }
    if (operation == "load_state") {
        DOSBoxMCPV2ReleaseAll();
        dosbox_mcp_v2_restore_pic = DOSBoxMCPV2PICUS();
        dosbox_mcp_v2_restore_frame = dosbox_mcp_v2_frame;
        dosbox_mcp_v2_restore_cycles = DOSBoxMCPV2Cycles();
        dosbox_mcp_v2_request_id = request;
        dosbox_mcp_v2_restore_wait = true;
        /* Some DOSBox-X state components resume through a non-local control
         * transfer, so every continuation marker must be armed first. */
        LoadGameState_Run();
        return;
    }
    if (operation == "observe_writes") {
        const uint64_t start = DOSBoxMCPV2Integer(json, "physical_start");
        const uint64_t length = DOSBoxMCPV2Integer(json, "length");
        const uint64_t duration = DOSBoxMCPV2Integer(json, "duration_counter");
        const uint64_t limit = DOSBoxMCPV2Integer(json, "hit_limit", 10000);
        dosbox_mcp_v2_watch_clock = DOSBoxMCPV2String(json, "clock");
        if (start > 0x10ffff || length == 0 || length > 65536 || start + length > 0x110000 || duration == 0 ||
            (dosbox_mcp_v2_watch_clock != "video_frame" && dosbox_mcp_v2_watch_clock != "pic_us" && dosbox_mcp_v2_watch_clock != "cpu_cycles")) {
            DOSBoxMCPV2Error(request, "invalid bounded write-watch request"); return;
        }
        dosbox_mcp_v2_request_id = request;
        dosbox_mcp_v2_watch_start = (uint32_t)start;
        dosbox_mcp_v2_watch_end = (uint32_t)(start + length);
        dosbox_mcp_v2_watch_until = DOSBoxMCPV2Counter(dosbox_mcp_v2_watch_clock) + duration;
        dosbox_mcp_v2_watch_limit = (size_t)std::min<uint64_t>(limit, 10000);
        dosbox_mcp_v2_writes.clear();
        dosbox_mcp_v2_writes_truncated = false;
        dosbox_mcp_write_watch_enabled = true;
        dosbox_mcp_v2_watch_in_timeline = false;
        return;
    }
    if (operation == "video_snapshot") {
        const uint64_t after = DOSBoxMCPV2Integer(json, "after_frames", 1);
        const uint64_t maximum = DOSBoxMCPV2Integer(json, "max_bytes", 1024 * 1024);
        if (after > 100000 || maximum == 0 || maximum > 1024 * 1024) {
            DOSBoxMCPV2Error(request, "invalid bounded video snapshot request"); return;
        }
        dosbox_mcp_v2_request_id = request;
        dosbox_mcp_v2_video_target = dosbox_mcp_v2_frame + after;
        dosbox_mcp_v2_video_limit = (size_t)maximum;
        dosbox_mcp_v2_video_wait = true;
        return;
    }
    if (operation != "run_timeline") { DOSBoxMCPV2Error(request, "unsupported operation"); return; }
    dosbox_mcp_v2_clock = DOSBoxMCPV2String(json, "clock");
    if (dosbox_mcp_v2_clock != "video_frame" && dosbox_mcp_v2_clock != "pic_us" &&
        dosbox_mcp_v2_clock != "cpu_cycles" && dosbox_mcp_v2_clock != "wall_clock") {
        DOSBoxMCPV2Error(request, "unsupported timeline clock"); return;
    }
    if (!DOSBoxMCPV2ParseTimeline(json)) { DOSBoxMCPV2Error(request, "malformed or empty timeline"); return; }
    dosbox_mcp_v2_request_id = request;
    dosbox_mcp_v2_start = DOSBoxMCPV2Counter(dosbox_mcp_v2_clock);
    dosbox_mcp_v2_expire_after = DOSBoxMCPV2Integer(json, "expire_after_counter");
    const uint64_t expire_pic_delta = DOSBoxMCPV2Integer(json, "expire_after_pic_us");
    dosbox_mcp_v2_expire_pic_us = expire_pic_delta ? DOSBoxMCPV2PICUS() + expire_pic_delta : 0;
    dosbox_mcp_v2_next = 0;
    dosbox_mcp_v2_active = true;
}

static bool DOSBoxMCPV2ReadRequest(std::string &message) {
#if defined(WIN32)
    if (dosbox_mcp_v2_pipe == INVALID_HANDLE_VALUE) return false;
    if (!dosbox_mcp_v2_connected) {
        const BOOL connected = ConnectNamedPipe(dosbox_mcp_v2_pipe, NULL);
        const DWORD error = GetLastError();
        if (!connected && error != ERROR_PIPE_CONNECTED) {
            if (error != ERROR_PIPE_LISTENING && error != ERROR_NO_DATA) return false;
            return false;
        }
        dosbox_mcp_v2_connected = true;
    }
    DWORD available = 0;
    if (!PeekNamedPipe(dosbox_mcp_v2_pipe, NULL, 0, NULL, &available, NULL)) {
        DOSBoxMCPV2ReleaseAll(); DOSBoxMCPV2CloseClient(); return false;
    }
    if (available) {
        std::vector<char> data(available);
        DWORD done = 0;
        if (!ReadFile(dosbox_mcp_v2_pipe, &data[0], available, &done, NULL)) {
            DOSBoxMCPV2ReleaseAll(); DOSBoxMCPV2CloseClient(); return false;
        }
        dosbox_mcp_v2_input.append(&data[0], done);
    }
#else
    if (dosbox_mcp_v2_server < 0) return false;
    if (dosbox_mcp_v2_client < 0) {
        dosbox_mcp_v2_client = accept(dosbox_mcp_v2_server, NULL, NULL);
        if (dosbox_mcp_v2_client < 0) return false;
        fcntl(dosbox_mcp_v2_client, F_SETFL, O_NONBLOCK);
    }
    char data[65536];
    const ssize_t done = recv(dosbox_mcp_v2_client, data, sizeof(data), MSG_DONTWAIT);
    if (done > 0) dosbox_mcp_v2_input.append(data, (size_t)done);
    else if (done == 0) { DOSBoxMCPV2ReleaseAll(); DOSBoxMCPV2CloseClient(); return false; }
#endif
    if (dosbox_mcp_v2_input.size() < 4) return false;
    const unsigned char *header = (const unsigned char*)dosbox_mcp_v2_input.data();
    const uint32_t size = (uint32_t)header[0] | ((uint32_t)header[1] << 8) | ((uint32_t)header[2] << 16) | ((uint32_t)header[3] << 24);
    if (size > 4 * 1024 * 1024) { DOSBoxMCPV2Error("", "message too large"); return false; }
    if (dosbox_mcp_v2_input.size() < (size_t)size + 4) return false;
    message.assign(dosbox_mcp_v2_input.data() + 4, size);
    dosbox_mcp_v2_input.erase(0, (size_t)size + 4);
    return true;
}

static void DOSBoxMCPV2FinishTimeline(void) {
    DOSBoxMCPV2ReleaseAll();
    dosbox_mcp_write_watch_enabled = false;
    dosbox_mcp_v2_watch_in_timeline = false;
    std::ostringstream timeline, events;
    timeline << "{\"timeline_events\":[";
    events << '[';
    for (size_t i = 0; i < dosbox_mcp_v2_timeline.size(); ++i) {
        const DOSBoxMCPV2TimelineEvent &event = dosbox_mcp_v2_timeline[i];
        if (i) { timeline << ','; events << ','; }
        timeline << "{\"order\":" << event.order << ",\"kind\":\"" << event.kind
                 << "\",\"requested_counter\":" << event.at << ",\"actual_counter\":" << event.actual
                 << ",\"acknowledgement\":\"" << event.acknowledgement << "\""
                 << ",\"irq1_observed\":" << ((event.kind == "key_make" || event.kind == "key_break" || event.kind == "key_set") && dosbox_mcp_v2_irq1_sequence > event.irq_before ? "true" : "false");
        if (!event.observation.empty()) timeline << ",\"observation\":" << event.observation;
        timeline << "}";
        events << "{\"session_id\":\"" << dosbox_mcp_v2_session << "\",\"sequence\":"
               << ++dosbox_mcp_v2_sequence << ",\"type\":\"" << event.acknowledgement
               << "\",\"counters\":{\"pic_us\":" << event.pic_us
               << ",\"video_frame\":" << event.video_frame
               << ",\"cpu_cycles\":" << event.cpu_cycles << "}}";
    }
    timeline << "],\"released_all_keys\":true}";
    events << ']';
    dosbox_mcp_v2_active = false;
    dosbox_mcp_v2_finish_wait = false;
    DOSBoxMCPV2Write(DOSBoxMCPV2Envelope(dosbox_mcp_v2_request_id, timeline.str(), events.str()));
}

static void DOSBoxMCPV2Pump(void) {
    if (dosbox_mcp_v2_token.empty()) return;
    if ((dosbox_mcp_v2_active || dosbox_mcp_v2_restore_wait || dosbox_mcp_write_watch_enabled || dosbox_mcp_v2_video_wait) && !DOSBoxMCPV2ClientAlive()) return;
    if (dosbox_mcp_v2_restore_wait && dosbox_mcp_v2_frame > dosbox_mcp_v2_restore_frame) {
        const std::string started_event = DOSBoxMCPV2Event("load_started", dosbox_mcp_v2_restore_pic, dosbox_mcp_v2_restore_frame, dosbox_mcp_v2_restore_cycles);
        const std::string completed_event = DOSBoxMCPV2Event("load_completed", DOSBoxMCPV2PICUS(), dosbox_mcp_v2_frame, DOSBoxMCPV2Cycles());
        const std::string frame_event = DOSBoxMCPV2Event("post_restore_frame", DOSBoxMCPV2PICUS(), dosbox_mcp_v2_frame, DOSBoxMCPV2Cycles());
        std::ostringstream events;
        events << '[' << started_event << ',' << completed_event << ',' << frame_event << ']';
        dosbox_mcp_v2_restore_wait = false;
        DOSBoxMCPV2Write(DOSBoxMCPV2Envelope(dosbox_mcp_v2_request_id, "{\"loaded\":true,\"post_restore_frame\":true}", events.str()));
    }
    if (dosbox_mcp_v2_restore_wait) return;
    if (dosbox_mcp_v2_video_wait && dosbox_mcp_v2_frame >= dosbox_mcp_v2_video_target) {
        const std::string observation = DOSBoxMCPV2VideoObservation(dosbox_mcp_v2_video_limit);
        const std::string event = DOSBoxMCPV2Event("video_snapshot", DOSBoxMCPV2PICUS(), dosbox_mcp_v2_frame, DOSBoxMCPV2Cycles());
        dosbox_mcp_v2_video_wait = false;
        DOSBoxMCPV2Write(DOSBoxMCPV2Envelope(dosbox_mcp_v2_request_id, observation, std::string("[") + event + "]"));
    }
    if (dosbox_mcp_v2_video_wait) return;
    if (dosbox_mcp_write_watch_enabled && !dosbox_mcp_v2_watch_in_timeline && DOSBoxMCPV2Counter(dosbox_mcp_v2_watch_clock) >= dosbox_mcp_v2_watch_until) {
        dosbox_mcp_write_watch_enabled = false;
        std::ostringstream result;
        result << "{\"write_events\":[";
        for (size_t i = 0; i < dosbox_mcp_v2_writes.size(); ++i) {
            const DOSBoxMCPV2WriteEvent &event = dosbox_mcp_v2_writes[i];
            if (i) result << ',';
            result << "{\"physical_address\":" << event.physical << ",\"width\":" << event.width
                   << ",\"before\":" << event.before << ",\"after\":" << event.after
                   << ",\"cs\":" << event.cs << ",\"ip\":" << event.ip << ",\"counter\":" << event.counter
                   << ",\"counters\":{\"pic_us\":" << event.pic_us << ",\"video_frame\":" << event.video_frame << ",\"cpu_cycles\":" << event.cpu_cycles << "}"
                   << ",\"registers\":{\"ax\":" << event.ax << ",\"bx\":" << event.bx << ",\"cx\":" << event.cx << ",\"dx\":" << event.dx
                   << ",\"si\":" << event.si << ",\"di\":" << event.di << ",\"bp\":" << event.bp << ",\"sp\":" << event.sp
                   << ",\"ds\":" << event.ds << ",\"es\":" << event.es << ",\"ss\":" << event.ss << "}}";
        }
        result << "],\"raw_hits\":" << dosbox_mcp_v2_writes.size() << ",\"truncated\":" << (dosbox_mcp_v2_writes_truncated ? "true" : "false") << "}";
        dosbox_mcp_v2_writes.clear();
        DOSBoxMCPV2Write(DOSBoxMCPV2Envelope(dosbox_mcp_v2_request_id, result.str()));
    }
    if (dosbox_mcp_write_watch_enabled && !dosbox_mcp_v2_watch_in_timeline) return;
    if (!dosbox_mcp_v2_active) {
        std::string request;
        if (DOSBoxMCPV2ReadRequest(request)) DOSBoxMCPV2Begin(request);
    }
    if (!dosbox_mcp_v2_active) return;
    if (dosbox_mcp_v2_finish_wait) {
        if (dosbox_mcp_v2_frame > dosbox_mcp_v2_finish_frame) DOSBoxMCPV2FinishTimeline();
        return;
    }
    const uint64_t now = DOSBoxMCPV2Counter(dosbox_mcp_v2_clock);
    if (((dosbox_mcp_v2_expire_after && now - dosbox_mcp_v2_start >= dosbox_mcp_v2_expire_after) ||
         (dosbox_mcp_v2_expire_pic_us && DOSBoxMCPV2PICUS() >= dosbox_mcp_v2_expire_pic_us)) &&
        dosbox_mcp_v2_next < dosbox_mcp_v2_timeline.size()) {
        DOSBoxMCPV2ReleaseAll();
        dosbox_mcp_write_watch_enabled = false;
        dosbox_mcp_v2_watch_in_timeline = false;
        dosbox_mcp_v2_active = false;
        dosbox_mcp_v2_finish_wait = false;
        DOSBoxMCPV2Error(dosbox_mcp_v2_request_id, "timeline expired before all events; all keys released");
        return;
    }
    while (dosbox_mcp_v2_next < dosbox_mcp_v2_timeline.size()) {
        DOSBoxMCPV2TimelineEvent &event = dosbox_mcp_v2_timeline[dosbox_mcp_v2_next];
        if (now - dosbox_mcp_v2_start < event.at) break;
        event.actual = now - dosbox_mcp_v2_start;
        event.pic_us = DOSBoxMCPV2PICUS();
        event.video_frame = dosbox_mcp_v2_frame;
        event.cpu_cycles = DOSBoxMCPV2Cycles();
        event.irq_before = dosbox_mcp_v2_irq1_sequence;
        if (event.kind == "key_make" && event.key > KBD_NONE && event.key < KBD_LAST) {
            KEYBOARD_AddKey((KBD_KEYS)event.key, true); dosbox_mcp_v2_pressed.insert(event.key); event.acknowledgement = "key_applied";
        } else if (event.kind == "key_break" && event.key > KBD_NONE && event.key < KBD_LAST) {
            KEYBOARD_AddKey((KBD_KEYS)event.key, false); dosbox_mcp_v2_pressed.erase(event.key); event.acknowledgement = "key_released";
        } else if (event.kind == "key_set") {
            std::set<int> wanted(event.keys.begin(), event.keys.end());
            for (std::set<int>::iterator it = dosbox_mcp_v2_pressed.begin(); it != dosbox_mcp_v2_pressed.end();) {
                const int key = *it++;
                if (!wanted.count(key)) { KEYBOARD_AddKey((KBD_KEYS)key, false); dosbox_mcp_v2_pressed.erase(key); }
            }
            for (std::set<int>::iterator it = wanted.begin(); it != wanted.end(); ++it)
                if (*it > KBD_NONE && *it < KBD_LAST && !dosbox_mcp_v2_pressed.count(*it)) { KEYBOARD_AddKey((KBD_KEYS)*it, true); dosbox_mcp_v2_pressed.insert(*it); }
            event.acknowledgement = "key_set_applied";
        } else if (event.kind == "capture_mark") {
            event.acknowledgement = "capture_mark";
            if (event.capture_bytes) event.observation = DOSBoxMCPV2VideoObservation(event.capture_bytes);
        }
        else if (event.kind == "memory_snapshot") {
            event.acknowledgement = "memory_snapshot_mark";
            if (event.length && event.length <= 65536 && event.physical <= 0x10ffff && event.physical + event.length <= 0x110000)
                event.observation = DOSBoxMCPV2MemoryObservation(event.physical, event.length);
        }
        else if (event.kind == "write_watch_start" && event.length && event.length <= 65536 && event.physical <= 0x10ffff && event.physical + event.length <= 0x110000 && event.hit_limit && event.hit_limit <= 1000 && !dosbox_mcp_write_watch_enabled) {
            dosbox_mcp_v2_watch_start = event.physical;
            dosbox_mcp_v2_watch_end = event.physical + event.length;
            dosbox_mcp_v2_watch_clock = dosbox_mcp_v2_clock;
            dosbox_mcp_v2_watch_until = ~(uint64_t)0;
            dosbox_mcp_v2_watch_limit = event.hit_limit;
            dosbox_mcp_v2_writes.clear();
            dosbox_mcp_v2_writes_truncated = false;
            dosbox_mcp_v2_watch_in_timeline = true;
            dosbox_mcp_write_watch_enabled = true;
            event.acknowledgement = "write_watch_started";
        }
        else if (event.kind == "write_watch_stop" && dosbox_mcp_write_watch_enabled && dosbox_mcp_v2_watch_in_timeline) {
            dosbox_mcp_write_watch_enabled = false;
            dosbox_mcp_v2_watch_in_timeline = false;
            std::ostringstream observed;
            observed << "{\"write_events\":[";
            for (size_t i = 0; i < dosbox_mcp_v2_writes.size(); ++i) {
                const DOSBoxMCPV2WriteEvent &hit = dosbox_mcp_v2_writes[i];
                if (i) observed << ',';
                observed << "{\"physical_address\":" << hit.physical << ",\"width\":" << hit.width
                         << ",\"before\":" << hit.before << ",\"after\":" << hit.after
                         << ",\"cs\":" << hit.cs << ",\"ip\":" << hit.ip << ",\"counter\":" << hit.counter
                         << ",\"counters\":{\"pic_us\":" << hit.pic_us << ",\"video_frame\":" << hit.video_frame << ",\"cpu_cycles\":" << hit.cpu_cycles << "}"
                         << ",\"registers\":{\"ax\":" << hit.ax << ",\"bx\":" << hit.bx << ",\"cx\":" << hit.cx << ",\"dx\":" << hit.dx
                         << ",\"si\":" << hit.si << ",\"di\":" << hit.di << ",\"bp\":" << hit.bp << ",\"sp\":" << hit.sp
                         << ",\"ds\":" << hit.ds << ",\"es\":" << hit.es << ",\"ss\":" << hit.ss << "}}";
            }
            observed << "],\"raw_hits\":" << dosbox_mcp_v2_writes.size() << ",\"truncated\":" << (dosbox_mcp_v2_writes_truncated ? "true" : "false") << "}";
            event.observation = observed.str();
            event.acknowledgement = "write_watch_stopped";
            dosbox_mcp_v2_writes.clear();
        }
        else if (event.kind == "trace_mark") event.acknowledgement = "trace_mark";
        else event.acknowledgement = "ignored_invalid_event";
        ++dosbox_mcp_v2_next;
    }
    if (dosbox_mcp_v2_next == dosbox_mcp_v2_timeline.size()) {
        dosbox_mcp_v2_finish_wait = true;
        dosbox_mcp_v2_finish_frame = dosbox_mcp_v2_frame;
    }
}

void DOSBoxMCPV2VideoFrame(void) {
    ++dosbox_mcp_v2_frame;
    /* A loaded state can replace the normal SDL update continuation. The VGA
     * retrace remains authoritative and guarantees the post-restore pump. */
    DOSBoxMCPV2Pump();
}

static void DOSBoxMCPV2Init(void) {
    const char *token = getenv("DOSBOX_X_MCP_TOKEN");
    if (token == NULL || *token == 0) return;
    dosbox_mcp_v2_token = token;
    std::ostringstream identity;
#if defined(WIN32)
    identity << std::hex << GetCurrentProcessId() << '-' << GetTickCount64();
    std::ostringstream endpoint;
    endpoint << "\\\\.\\pipe\\dosbox-x-mcp-" << GetCurrentProcessId();
    dosbox_mcp_v2_pipe = CreateNamedPipeA(endpoint.str().c_str(),
        PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_NOWAIT | PIPE_REJECT_REMOTE_CLIENTS,
        1, 4 * 1024 * 1024, 4 * 1024 * 1024, 0, NULL);
    if (dosbox_mcp_v2_pipe == INVALID_HANDLE_VALUE) { dosbox_mcp_v2_token.clear(); return; }
#else
    identity << std::hex << getpid() << '-' << (uint64_t)SDL_GetTicks();
    const char *runtime = getenv("XDG_RUNTIME_DIR");
    std::ostringstream directory;
    if (runtime && *runtime) directory << runtime;
    else directory << "/tmp/dosbox-x-mcp-" << getuid();
    mkdir(directory.str().c_str(), 0700);
    chmod(directory.str().c_str(), 0700);
    std::ostringstream endpoint;
    endpoint << directory.str() << "/dosbox-x-mcp-" << getpid() << ".sock";
    dosbox_mcp_v2_socket_path = endpoint.str();
    unlink(dosbox_mcp_v2_socket_path.c_str());
    dosbox_mcp_v2_server = socket(AF_UNIX, SOCK_STREAM, 0);
    if (dosbox_mcp_v2_server < 0) { dosbox_mcp_v2_token.clear(); return; }
    sockaddr_un address; memset(&address, 0, sizeof(address)); address.sun_family = AF_UNIX;
    strncpy(address.sun_path, dosbox_mcp_v2_socket_path.c_str(), sizeof(address.sun_path) - 1);
    if (bind(dosbox_mcp_v2_server, (sockaddr*)&address, sizeof(address)) != 0 || listen(dosbox_mcp_v2_server, 1) != 0) {
        close(dosbox_mcp_v2_server); dosbox_mcp_v2_server = -1; dosbox_mcp_v2_token.clear(); return;
    }
    chmod(dosbox_mcp_v2_socket_path.c_str(), 0600);
    fcntl(dosbox_mcp_v2_server, F_SETFL, O_NONBLOCK);
#endif
    dosbox_mcp_v2_session = identity.str();
}
