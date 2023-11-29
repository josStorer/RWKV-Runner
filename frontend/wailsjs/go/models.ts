export namespace backend_golang {
	
	export class FileInfo {
	    name: string;
	    size: number;
	    isDir: boolean;
	    modTime: string;
	
	    static createFrom(source: any = {}) {
	        return new FileInfo(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.name = source["name"];
	        this.size = source["size"];
	        this.isDir = source["isDir"];
	        this.modTime = source["modTime"];
	    }
	}
	export class MIDIMessage {
	    messageType: string;
	    channel: number;
	    note: number;
	    velocity: number;
	    control: number;
	    value: number;
	
	    static createFrom(source: any = {}) {
	        return new MIDIMessage(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.messageType = source["messageType"];
	        this.channel = source["channel"];
	        this.note = source["note"];
	        this.velocity = source["velocity"];
	        this.control = source["control"];
	        this.value = source["value"];
	    }
	}

}

