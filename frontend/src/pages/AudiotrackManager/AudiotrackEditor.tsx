import React, { FC, useEffect, useRef, useState } from 'react';
import { observer } from 'mobx-react-lite';
import { useTranslation } from 'react-i18next';
import Draggable from 'react-draggable';
import { ToolTipButton } from '../../components/ToolTipButton';
import { v4 as uuid } from 'uuid';
import {
  Add16Regular,
  ArrowAutofitWidth20Regular,
  Delete16Regular,
  MusicNote220Regular,
  Play16Regular,
  Record16Regular
} from '@fluentui/react-icons';
import { Button, Card, Slider, Text, Tooltip } from '@fluentui/react-components';
import { useWindowSize } from 'usehooks-ts';
import commonStore from '../../stores/commonStore';
import classnames from 'classnames';

const snapValue = 25;
const minimalMoveTime = 8; // 1000/125=8ms
const scaleMin = 0.2;
const scaleMax = 3;
const baseMoveTime = Math.round(minimalMoveTime / scaleMin);

type TrackProps = {
  id: string;
  right: number;
  scale: number;
  isSelected: boolean;
  onSelect: (id: string) => void;
};

const Track: React.FC<TrackProps> = observer(({
  id,
  right,
  scale,
  isSelected,
  onSelect
}) => {
  const { t } = useTranslation();
  const trackIndex = commonStore.tracks.findIndex(t => t.id === id)!;
  const track = commonStore.tracks[trackIndex];
  const trackClass = isSelected ? 'bg-blue-600' : 'bg-gray-700';
  const controlX = useRef(0);

  return (
    <Draggable
      axis="x"
      bounds={{ left: 0, right }}
      grid={[snapValue, snapValue]}
      position={{
        x: (track.offsetTime - commonStore.trackCurrentTime) / (baseMoveTime * scale) * snapValue,
        y: 0
      }}
      onStart={(e, data) => {
        controlX.current = data.lastX;
      }}
      onStop={(e, data) => {
        const delta = data.lastX - controlX.current;
        let offsetTime = Math.round(Math.round(delta / snapValue * baseMoveTime * scale) / minimalMoveTime) * minimalMoveTime;
        offsetTime = Math.min(Math.max(
          offsetTime,
          -track.offsetTime), commonStore.trackTotalTime - track.offsetTime);

        const tracks = commonStore.tracks.slice();
        tracks[trackIndex].offsetTime += offsetTime;
        commonStore.setTracks(tracks);
      }}
    >
      <div
        className={`p-1 cursor-move rounded whitespace-nowrap overflow-hidden ${trackClass}`}
        style={{
          width: `${Math.max(80,
            track.contentTime / (baseMoveTime * scale) * snapValue
          )}px`
        }}
        onClick={() => onSelect(id)}
      >
        <span className="text-white">{t('Track') + ' ' + id}</span>
      </div>
    </Draggable>
  );
});

const AudiotrackEditor: FC = observer(() => {
  const { t } = useTranslation();

  const currentTimeControlRef = useRef<HTMLDivElement>(null);
  const playStartTimeControlRef = useRef<HTMLDivElement>(null);
  const tracksRef = useRef<HTMLDivElement>(null);
  const toolbarRef = useRef<HTMLDivElement>(null);
  const toolbarButtonRef = useRef<HTMLDivElement>(null);
  const toolbarSliderRef = useRef<HTMLInputElement>(null);

  const [refreshRef, setRefreshRef] = useState(false);

  const windowSize = useWindowSize();
  const scale = (scaleMin + scaleMax) - commonStore.trackScale;

  const [selectedTrackId, setSelectedTrackId] = useState<string>('');
  const playStartTimeControlX = useRef(0);
  const selectedTrack = selectedTrackId ? commonStore.tracks.find(t => t.id === selectedTrackId) : undefined;

  useEffect(() => {
    setRefreshRef(!refreshRef);
  }, [windowSize, commonStore.tracks]);

  const viewControlsContainerWidth = (toolbarRef.current && toolbarButtonRef.current && toolbarSliderRef.current) ?
    toolbarRef.current.clientWidth - toolbarButtonRef.current.clientWidth - toolbarSliderRef.current.clientWidth - 16 // 16 = ml-2 mr-2
    : 0;
  const tracksWidth = viewControlsContainerWidth;
  const timeOfTracksWidth = Math.floor(tracksWidth / snapValue) // number of moves
    * baseMoveTime * scale;
  const currentTimeControlWidth = (timeOfTracksWidth < commonStore.trackTotalTime)
    ? timeOfTracksWidth / commonStore.trackTotalTime * viewControlsContainerWidth
    : 0;
  const playStartTimeControlPosition = {
    x: (commonStore.trackPlayStartTime - commonStore.trackCurrentTime) / (baseMoveTime * scale) * snapValue,
    y: 0
  };

  return (
    <div className="flex flex-col gap-2 overflow-hidden" style={{ width: '80vw', height: '80vh' }}>
      <div className="mx-auto">
        <Text size={100}>{`${commonStore.trackPlayStartTime} ms / ${commonStore.trackTotalTime} ms`}</Text>
      </div>
      <div className="flex pb-2 border-b" ref={toolbarRef}>
        <div className="flex gap-2" ref={toolbarButtonRef}>
          <ToolTipButton desc={t('Play All')} icon={<Play16Regular />} />
          <ToolTipButton desc={t('Clear All')} icon={<Delete16Regular />} onClick={() => {
            commonStore.setTracks([]);
          }} />
        </div>
        <div className="grow">
          <div className="flex flex-col ml-2 mr-2">
            <Draggable axis="x" bounds={{
              left: 0,
              right: viewControlsContainerWidth - currentTimeControlWidth
            }}
              position={{
                x: commonStore.trackCurrentTime / commonStore.trackTotalTime * viewControlsContainerWidth,
                y: 0
              }}
              onDrag={(e, data) => {
                setTimeout(() => {
                  let offset = 0;
                  if (currentTimeControlRef.current) {
                    const match = currentTimeControlRef.current.style.transform.match(/translate\((.+)px,/);
                    if (match)
                      offset = parseFloat(match[1]);
                  }
                  const offsetTime = commonStore.trackTotalTime / viewControlsContainerWidth * offset;
                  commonStore.setTrackCurrentTime(offsetTime);
                }, 1);
              }}
            >
              <div ref={currentTimeControlRef} className="h-2 bg-gray-700 cursor-move rounded"
                style={{ width: currentTimeControlWidth }} />
            </Draggable>
            <div className={classnames(
              'flex',
              (playStartTimeControlPosition.x < 0 || playStartTimeControlPosition.x > viewControlsContainerWidth)
              && 'hidden'
            )}>
              <Draggable axis="x" bounds={{
                left: 0,
                right: (playStartTimeControlRef.current)
                  ? viewControlsContainerWidth - playStartTimeControlRef.current.clientWidth
                  : 0
              }}
                grid={[snapValue, snapValue]}
                position={playStartTimeControlPosition}
                onStart={(e, data) => {
                  playStartTimeControlX.current = data.lastX;
                }}
                onStop={(e, data) => {
                  const delta = data.lastX - playStartTimeControlX.current;
                  let offsetTime = Math.round(Math.round(delta / snapValue * baseMoveTime * scale) / minimalMoveTime) * minimalMoveTime;
                  offsetTime = Math.min(Math.max(
                    offsetTime,
                    -commonStore.trackPlayStartTime), commonStore.trackTotalTime - commonStore.trackPlayStartTime);
                  commonStore.setTrackPlayStartTime(commonStore.trackPlayStartTime + offsetTime);
                }}
              >
                <div className="relative cursor-move"
                  ref={playStartTimeControlRef}>
                  <ArrowAutofitWidth20Regular />
                  <div className="border-l absolute border-gray-700"
                    style={{
                      height: (tracksRef.current && commonStore.tracks.length > 0)
                        ? tracksRef.current.clientHeight
                        : 0,
                      top: '50%',
                      left: 'calc(50% - 0.5px)'
                    }} />
                </div>
              </Draggable>
            </div>
          </div>
        </div>
        <Tooltip content={t('Scale View')!} showDelay={0} hideDelay={0} relationship="label">
          <Slider ref={toolbarSliderRef} value={commonStore.trackScale} step={scaleMin} max={scaleMax} min={scaleMin}
            onChange={(e, data) => {
              commonStore.setTrackScale(data.value);
            }}
          />
        </Tooltip>
      </div>
      <div className="flex flex-col overflow-y-auto gap-1" ref={tracksRef}>
        {commonStore.tracks.map(track =>
          <div key={track.id} className="flex gap-2 pb-1 border-b">
            <div className="flex gap-1 border-r h-7">
              <ToolTipButton desc={t('Record')} icon={<Record16Regular />} size="small" shape="circular"
                appearance="subtle" />
              <ToolTipButton desc={t('Play')} icon={<Play16Regular />} size="small" shape="circular"
                appearance="subtle" />
              <ToolTipButton desc={t('Delete')} icon={<Delete16Regular />} size="small" shape="circular"
                appearance="subtle" onClick={() => {
                const tracks = commonStore.tracks.slice().filter(t => t.id !== track.id);
                commonStore.setTracks(tracks);
              }} />
            </div>
            <div className="relative grow overflow-hidden">
              <div className="absolute" style={{ left: -0 }}>
                <Track
                  id={track.id}
                  scale={scale}
                  right={tracksWidth}
                  isSelected={selectedTrackId === track.id}
                  onSelect={setSelectedTrackId}
                />
              </div>
            </div>
          </div>)}
        <div className="flex justify-between items-center">
          <Button icon={<Add16Regular />} size="small" shape="circular"
            appearance="subtle"
            onClick={() => {
              commonStore.setTracks([...commonStore.tracks, {
                id: uuid(),
                content: '',
                offsetTime: 0,
                contentTime: 0
              }]);
            }}>
            {t('New Track')}
          </Button>
          <Text size={100}>
            {t('Select a track to preview the content')}
          </Text>
        </div>
      </div>
      <div className="grow"></div>
      {selectedTrack &&
        <Card size="small" appearance="outline" style={{ minHeight: '150px' }}>
          <div className="flex flex-col gap-1 overflow-hidden">
            <Text size={100}>{`${t('Start Time')}: ${selectedTrack.offsetTime} ms`}</Text>
            <Text size={100}>{`${t('Content Time')}: ${selectedTrack.contentTime} ms`}</Text>
            <div className="overflow-y-auto overflow-x-hidden">
              {selectedTrack.content}
            </div>
          </div>
        </Card>
      }
      <Button icon={<MusicNote220Regular />} style={{ minHeight: '32px' }}>
        {t('Save to generation area')}
      </Button>
    </div>
  );
});

export default AudiotrackEditor;
