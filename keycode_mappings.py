import evdev
from evdev import ecodes
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DIGITS = "0123456789"
# _CHAR is the dictionary of key to unicode mappings for alphabets
_CHAR = {}
if evdev is not None:
    for _c in _LETTERS:
        _CHAR[getattr(ecodes, f"KEY_{_c}")] = ord(_c)
    for _d in _DIGITS:
        _CHAR[getattr(ecodes, f"KEY_{_d}")] = ord(_d)

# _GLFW_SPECIAL is the dictionary of key to unicode mappings for special keys like 'space'
_GLFW_SPECIAL = {
    ecodes.KEY_SPACE: 32, ecodes.KEY_ENTER: 257, ecodes.KEY_TAB: 258,
    ecodes.KEY_BACKSPACE: 259, ecodes.KEY_INSERT: 260, ecodes.KEY_DELETE: 261,
    ecodes.KEY_RIGHT: 262, ecodes.KEY_LEFT: 263, ecodes.KEY_DOWN: 264,
    ecodes.KEY_UP: 265, ecodes.KEY_PAGEUP: 266, ecodes.KEY_PAGEDOWN: 267,
    ecodes.KEY_HOME: 268, ecodes.KEY_END: 269, ecodes.KEY_CAPSLOCK: 280,
    ecodes.KEY_SCROLLLOCK: 281, ecodes.KEY_NUMLOCK: 282, ecodes.KEY_SYSRQ: 283,
    ecodes.KEY_PAUSE: 284, ecodes.KEY_F1: 290, ecodes.KEY_F2: 291,
    ecodes.KEY_F3: 292, ecodes.KEY_F4: 293, ecodes.KEY_F5: 294, ecodes.KEY_F6: 295,
    ecodes.KEY_F7: 296, ecodes.KEY_F8: 297, ecodes.KEY_F9: 298, ecodes.KEY_F10: 299,
    ecodes.KEY_F11: 300, ecodes.KEY_F12: 301, ecodes.KEY_LEFTSHIFT: 340,
    ecodes.KEY_RIGHTSHIFT: 344, ecodes.KEY_LEFTCTRL: 341, ecodes.KEY_RIGHTCTRL: 345,
    ecodes.KEY_LEFTALT: 342, ecodes.KEY_RIGHTALT: 346, ecodes.KEY_ESC: 256,
} if evdev is not None else {}

# _CLICK_BTN is the dictionary of mouse click types
CLICK_BTN = {
    ecodes.BTN_LEFT:   "LEFT",
    ecodes.BTN_RIGHT:  "RIGHT",
    ecodes.BTN_MIDDLE: "MIDDLE",
} if evdev is not None else {}


def glfw_code(keycode):
    """Returns the correct unicode mapping for keycode."""
    if keycode in _GLFW_SPECIAL: #check in for special first
        return _GLFW_SPECIAL[keycode]
    if keycode in _CHAR:    # then check for normal characters
        return _CHAR[keycode]
    return -1   # unmapped; matches _key_code's fallback