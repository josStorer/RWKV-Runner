package backend_golang

import (
	"errors"
	"fmt"
	"time"

	"github.com/mattrtaylor/go-rtmidi"
	"github.com/wailsapp/wails/v2/pkg/runtime"
)

type Port struct {
	Name string `json:"name"`
}
type MIDIMessage struct {
	MessageType string `json:"messageType"`
	Channel     int    `json:"channel"`
	Note        int    `json:"note"`
	Velocity    int    `json:"velocity"`
	Control     int    `json:"control"`
	Value       int    `json:"value"`
}

var ports []Port
var input rtmidi.MIDIIn
var out rtmidi.MIDIOut
var activeIndex int = -1
var lastNoteTime time.Time

func (a *App) midiLoop() {
	var err error
	input, err = rtmidi.NewMIDIInDefault()
	if err != nil {
		runtime.EventsEmit(a.ctx, "midiError", err.Error())
		return
	}
	out, err = rtmidi.NewMIDIOutDefault()
	if err != nil {
		runtime.EventsEmit(a.ctx, "midiError", err.Error())
	}
	err = out.OpenPort(0, "")
	if err != nil {
		runtime.EventsEmit(a.ctx, "midiError", err.Error())
	}
	ticker := time.NewTicker(500 * time.Millisecond)
	go func() {
		for {
			<-ticker.C
			count, err := input.PortCount()
			if err != nil {
				continue
			}
			ports = make([]Port, count)
			for i := 0; i < count; i++ {
				name, err := input.PortName(i)
				if err == nil {
					ports[i].Name = name
				}
			}
			runtime.EventsEmit(a.ctx, "midiPorts", &ports)
		}
	}()
}

func (a *App) OpenMidiPort(index int) error {
	if input == nil {
		return errors.New("failed to initialize MIDI input")
	}
	if activeIndex == index {
		return nil
	}
	input.Destroy()
	var err error
	input, err = rtmidi.NewMIDIInDefault()
	if err != nil {
		return err
	}
	err = input.SetCallback(func(msg rtmidi.MIDIIn, bytes []byte, t float64) {
		// https://www.midi.org/specifications-old/item/table-1-summary-of-midi-message
		// https://www.rfc-editor.org/rfc/rfc6295.html
		//
		// msgType channel
		//  1001     0000
		//
		msgType := bytes[0] >> 4
		channel := bytes[0] & 0x0f
		switch msgType {
		case 0x8:
			elapsed := time.Since(lastNoteTime)
			lastNoteTime = time.Now()
			runtime.EventsEmit(a.ctx, "midiMessage", &MIDIMessage{
				MessageType: "ElapsedTime",
				Value:       int(elapsed.Milliseconds()),
			})
			note := bytes[1]
			runtime.EventsEmit(a.ctx, "midiMessage", &MIDIMessage{
				MessageType: "NoteOff",
				Channel:     int(channel),
				Note:        int(note),
			})
		case 0x9:
			elapsed := time.Since(lastNoteTime)
			lastNoteTime = time.Now()
			runtime.EventsEmit(a.ctx, "midiMessage", &MIDIMessage{
				MessageType: "ElapsedTime",
				Value:       int(elapsed.Milliseconds()),
			})
			note := bytes[1]
			velocity := bytes[2]
			runtime.EventsEmit(a.ctx, "midiMessage", &MIDIMessage{
				MessageType: "NoteOn",
				Channel:     int(channel),
				Note:        int(note),
				Velocity:    int(velocity),
			})
		case 0xb:
			// control 12 => K1 knob, control 13 => K2 knob
			control := bytes[1]
			value := bytes[2]
			runtime.EventsEmit(a.ctx, "midiMessage", &MIDIMessage{
				MessageType: "ControlChange",
				Channel:     int(channel),
				Control:     int(control),
				Value:       int(value),
			})
		default:
			fmt.Printf("Unknown midi message: %v\n", bytes)
		}
	})
	if err != nil {
		return err
	}
	err = input.OpenPort(index, "")
	if err != nil {
		return err
	}
	activeIndex = index
	lastNoteTime = time.Now()
	return nil
}

func (a *App) CloseMidiPort() error {
	if input == nil {
		return errors.New("failed to initialize MIDI input")
	}
	if activeIndex == -1 {
		return nil
	}
	activeIndex = -1
	input.Destroy()
	var err error
	input, err = rtmidi.NewMIDIInDefault()
	if err != nil {
		return err
	}
	return nil
}

func (a *App) PlayNote(msg MIDIMessage) error {
	if out == nil {
		return errors.New("failed to initialize MIDI output")
	}
	channelByte := byte(msg.Channel)
	if msg.MessageType == "NoteOn" {
		out.SendMessage([]byte{0x90 | channelByte, byte(msg.Note), byte(msg.Velocity)})
	} else if msg.MessageType == "NoteOff" {
		out.SendMessage([]byte{0x80 | channelByte, byte(msg.Note), byte(msg.Velocity)})
	}
	return nil
}
