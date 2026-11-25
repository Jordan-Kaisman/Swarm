(() => {
  "use strict";
  const byId = id => document.getElementById(id);
  const replay = JSON.parse(byId("replay-data").textContent);
  const { frames, width, height, dronesPerTeam, basesPerTeam } = replay;
  const lastTurn = frames.length - 1;
  const arena = byId("arena");
  const picker = byId("entity");
  const slider = byId("turn");
  const phaseButtons = [...document.querySelectorAll("[data-phase]")];
  const tracks = [...document.querySelectorAll(".event-track")];
  const teamNames = ["Blue", "Red"];
  const teamClasses = ["blue", "red"];
  const outcomeCauses = {
    bases_destroyed: "all bases destroyed",
    elimination: "all drones eliminated",
    mutual_elimination: "mutual elimination",
    simultaneous_knockout: "simultaneous knockout",
    truncation: "step limit reached",
  };
  const numberFormat = new Intl.NumberFormat("en", { maximumFractionDigits: 2, useGrouping: false });
  const scoreFormat = new Intl.NumberFormat("en", { maximumFractionDigits: 4, signDisplay: "exceptZero" });
  const number = value => numberFormat.format(Math.abs(value) < 0.005 ? 0 : value);
  const quantity = (value, unit) => {
    const numeral = number(value);
    return `${numeral} ${unit}${numeral === "1" ? "" : "s"}`;
  };
  const score = value => value == null ? "—" : scoreFormat.format(value);
  const coord = ([x, y]) => `(${number(x)}, ${number(y)})`;
  const degrees = radians => radians * 180 / Math.PI;
  const compass = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"];
  const headingLabel = radians => `${number(degrees(radians))}° ${compass[(Math.round(radians * 4 / Math.PI) + 8) % 8]}`;
  const length = vector => Math.hypot(...vector);
  const add = ([x, y], [dx, dy]) => [x + dx, y + dy];
  const size = { width: 0, height: 0 };
  let turn = Math.min(1, lastTurn);
  let phase = "after";
  let selected = "drone-0";
  let timer = null;

  const entities = new Map();
  for (const [kind, perTeam] of [["drone", dronesPerTeam], ["base", basesPerTeam]]) {
    for (let index = 0; index < 2 * perTeam; index++) {
      const team = Math.floor(index / perTeam);
      const slot = index % perTeam + 1;
      const key = `${kind}-${index}`;
      const name = `${teamNames[team]} ${kind} ${slot}`;
      const label = `${teamNames[team][0]}${kind === "base" ? "B" : ""}${slot}`;
      entities.set(key, { kind, index, team, key, name, label });
    }
  }

  const position = (entity, frame) => entity.kind === "drone" ? frame.pos[entity.index] : replay.basePos[entity.index];
  const alive = (entity, frame) => entity.kind === "drone" ? frame.alive[entity.index] : frame.hp[entity.index] > 0;
  const entityAt = (kind, index) => entities.get(`${kind}-${index}`);

  for (const [team, teamName] of teamNames.entries()) {
    const group = document.createElement("optgroup");
    group.label = teamName;
    for (const entity of entities.values()) {
      if (entity.team === team) group.append(new Option(`${entity.name} (${entity.label})`, entity.key));
    }
    picker.append(group);
  }
  byId("blue-policy").textContent = `Blue · ${replay.policies[0]}`;
  byId("red-policy").textContent = `Red · ${replay.policies[1]}`;
  byId("match-meta").textContent = [
    replay.seed == null ? "" : `Seed ${replay.seed}, game 1`,
    quantity(frames[lastTurn].t, "turn"),
    `${number(width)} × ${number(height)} arena`,
  ].filter(Boolean).join(" · ");
  const outcome = replay.cause === "truncation" ? "No knockout" :
    replay.winner === null ? "Draw" : `${teamNames[replay.winner]} wins`;
  byId("outcome").textContent = `${outcome} · ${outcomeCauses[replay.cause] ?? replay.cause}`;
  byId("final-return").textContent = score(replay.finalReturn);

  function eventsFor(frame) {
    const damage = (frame.basesHit ?? []).flatMap((hits, index) => {
      if (!hits) return [];
      const base = entityAt("base", index);
      return [frame.basesDestroyed[index] ? `${base.label} destroyed` : `${base.label} damaged`];
    });
    const elimination = (frame.dronesKilled ?? []).flatMap((killed, index) =>
      killed ? [`${entityAt("drone", index).label} eliminated`] : []);
    return { damage, elimination, finish: frame === frames[lastTurn] ? [outcome] : [] };
  }
  const events = frames.map(eventsFor);
  const eventTurns = events.flatMap((event, index) => Object.values(event).some(list => list.length) ? [index] : []);
  const eventPosition = index => `${100 * index / Math.max(lastTurn, 1)}%`;
  const eventCursors = [];
  for (const track of tracks) {
    for (const index of eventTurns) {
      const labels = events[index][track.dataset.event];
      if (!labels.length) continue;
      const marker = document.createElement("button");
      marker.type = "button";
      marker.className = "event-marker";
      marker.dataset.turn = index;
      marker.title = `Turn ${frames[index].t} · ${labels.join(", ")}`;
      marker.setAttribute("aria-label", marker.title);
      marker.style.setProperty("--position", eventPosition(index));
      marker.addEventListener("click", () => seek(index));
      track.append(marker);
    }
    const cursor = document.createElement("span");
    cursor.className = "event-cursor";
    cursor.setAttribute("aria-hidden", "true");
    track.append(cursor);
    eventCursors.push(cursor);
  }
  const eventMarkers = [...document.querySelectorAll(".event-marker")];

  function svg(tag, attributes, parent) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (parent) parent.append(node);
    return node;
  }

  function line(parent, start, end, className, marker) {
    const node = svg("line", {
      x1: start[0], y1: start[1], x2: end[0], y2: end[1], class: className,
    }, parent);
    if (marker) node.setAttribute("marker-end", `url(#${marker})`);
    return node;
  }

  function frameProjection() {
    const bounds = { x: 33, y: 22 };
    const scale = Math.min((size.width - 2 * bounds.x) / width, (size.height - 2 * bounds.y) / height);
    const left = (size.width - width * scale) / 2;
    const top = (size.height - height * scale) / 2;
    return { scale, left, top, point: ([x, y]) => [left + x * scale, top + (height - y) * scale] };
  }

  function renderGrid(projection) {
    const { scale, left, top, point } = projection;
    const grid = byId("arena-grid");
    const arenaWidth = width * scale;
    const arenaHeight = height * scale;
    const bounds = { x: left, y: top, width: arenaWidth, height: arenaHeight };
    for (const [index, team] of teamClasses.entries()) {
      svg("rect", { ...bounds, x: left + index * arenaWidth / 2, width: arenaWidth / 2,
        class: `arena-home ${team}` }, grid);
    }
    const targetSpacing = 68 / scale;
    const decade = 10 ** Math.floor(Math.log10(targetSpacing));
    const step = ([1, 2, 5, 10].find(value => value * decade >= targetSpacing) ?? 10) * decade;
    for (let x = step; x < width - step / 4; x += step) {
      line(grid, point([x, 0]), point([x, height]), "arena-grid-line");
    }
    for (let y = step; y < height - step / 4; y += step) {
      line(grid, point([0, y]), point([width, y]), "arena-grid-line");
    }
    line(grid, point([width / 2, 0]), point([width / 2, height]), "arena-midline");
    svg("rect", { ...bounds, class: "arena-boundary" }, grid);
    svg("rect", bounds, document.querySelector("#arena-clip"));
    for (const [text, [x, y], anchor] of [
      ["0", [left, top + arenaHeight + 15], "middle"],
      [`x ${number(width)}`, [left + arenaWidth, top + arenaHeight + 15], "end"],
      [`y ${number(height)}`, [left, top - 6], "start"],
    ]) {
      svg("text", { x, y, class: "arena-axis", "text-anchor": anchor }, grid).textContent = text;
    }
  }

  function renderCone(layer, entity, frame, projection) {
    const { index, team } = entity;
    const [x, y] = projection.point(frame.pos[index]);
    const heading = frame.heading[index];
    const radius = replay.range[index] * projection.scale;
    const angle = replay.cone[index];
    const className = `cone ${teamClasses[team]}`;
    if (angle >= 2 * Math.PI - 1e-9) {
      svg("circle", { cx: x, cy: y, r: radius, class: className }, layer);
      return;
    }
    const at = a => [x + radius * Math.cos(a), y - radius * Math.sin(a)].join(" ");
    const d = `M${x} ${y}L${at(heading - angle / 2)}A${radius} ${radius} 0 ${angle > Math.PI ? 1 : 0} 0 ${at(heading + angle / 2)}Z`;
    svg("path", { d, class: className }, layer);
  }

  function movement(entity, before, after) {
    if (!before || entity.kind !== "drone" || !alive(entity, before)) return null;
    const { index } = entity;
    const action = after.actions?.[index];
    const requested = action ? replay.moveVectors[action[0]] : null;
    const actual = after.disp?.[index];
    const clipped = requested && actual && length([requested[0] - actual[0], requested[1] - actual[1]]) > 1e-8;
    return { action, requested, actual, clipped };
  }

  function renderMovement(layer, entity, before, after, point, visiblePhase) {
    const move = movement(entity, before, after);
    if (!move || visiblePhase === "before") return;
    const start = position(entity, before);
    const origin = point(start);
    if (move.requested && length(move.requested) > 0 && (visiblePhase === "action" || move.clipped)) {
      line(layer, origin, point(add(start, move.requested)), "move-requested", "arrow-requested");
    }
    if (visiblePhase === "action" && move.action && move.action[1] !== 1) {
      const angle = after.heading[entity.index];
      const tip = [origin[0] + 23 * Math.cos(angle), origin[1] - 23 * Math.sin(angle)];
      line(layer, origin, tip, "heading-requested", "arrow-requested");
    }
    if (visiblePhase === "after" && move.actual && length(move.actual) > 1e-8) {
      line(layer, origin, point(position(entity, after)), "move-actual", "arrow-actual");
      svg("circle", { cx: origin[0], cy: origin[1], r: 3, class: "move-origin" }, layer);
    }
  }

  function renderShots(layer, frame, point) {
    for (const [shooter, kind, target, , hit] of frame.shots ?? []) {
      const from = entityAt("drone", shooter);
      const to = entityAt(kind, target);
      const related = selected === from.key || selected === to.key;
      const className = `shot ${teamClasses[from.team]}${hit ? "" : " miss"}${related ? " related" : ""}`;
      line(layer, point(position(from, frame)), point(position(to, frame)), className);
    }
  }

  function renderEntities(layer, frame, point) {
    const ordered = [...entities.values()].sort((a, b) => Number(a.key === selected) - Number(b.key === selected));
    const located = ordered.map(entity => ({ entity, xy: point(position(entity, frame)) }));
    for (const { entity, xy: [x, y] } of located) {
      const { kind, index, team, key } = entity;
      const active = alive(entity, frame);
      const group = svg("g", {
        class: `entity ${teamClasses[team]}${active ? "" : " inactive"}${key === selected ? " selected" : ""}`,
        transform: `translate(${x} ${y})`,
        "data-entity": key,
      }, layer);
      svg("title", {}, group).textContent = entity.name;
      svg("circle", { r: 13, class: "entity-hitbox" }, group);
      svg("circle", { r: 11, class: "entity-ring" }, group);
      if (!active) {
        svg("path", { d: "M-4 -4L4 4M-4 4L4 -4", class: "entity-cross" }, group);
      } else if (kind === "drone") {
        svg("path", { d: "M9 0L-6 -5L-3 0L-6 5Z", class: "entity-drone",
          transform: `rotate(${-degrees(frame.heading[index])})` }, group);
      } else {
        svg("rect", { x: -6, y: -6, width: 12, height: 12, rx: 1, class: "entity-base" }, group);
      }
    }

    const placed = [];
    const labels = svg("g", {}, layer);
    for (const { entity, xy: [x, y] } of located.reverse()) {
      const { kind, index, label, key } = entity;
      const data = kind === "base" ? `${frame.hp[index]} HP · ${quantity(replay.baseValue[index], "pt")}` : "";
      const w = data ? Math.max(67, data.length * 5.6) : Math.max(17, label.length * 7);
      const h = data ? 27 : 14;
      const candidates = [[x + 14, y - 7], [x - w - 14, y - 7], [x - w / 2, y + 13], [x - w / 2, y - h - 13]];
      const positions = candidates.map(([left, top]) => ({
        left: Math.max(3, Math.min(left, size.width - w - 3)),
        top: Math.max(2, Math.min(top, size.height - h - 2)),
      }));
      const { left, top } = positions.find(box =>
        !placed.some(other => box.left < other.right && box.left + w > other.left &&
          box.top < other.bottom && box.top + h > other.top) &&
        !located.some(({ xy: [cx, cy] }) => cx + 8 > box.left && cx - 8 < box.left + w &&
          cy + 8 > box.top && cy - 8 < box.top + h)
      ) ?? positions[0];
      placed.push({ left, top, right: left + w, bottom: top + h });
      const className = alive(entity, frame) ? "entity-label" : "entity-label inactive";
      const group = svg("g", { "data-entity": key }, labels);
      svg("text", { x: left, y: top + 11, class: className }, group).textContent = label;
      if (data) svg("text", { x: left, y: top + 23, class: "base-data" }, group).textContent = data;
    }
  }

  function renderArena() {
    const after = frames[turn];
    const before = frames[turn - 1];
    const visiblePhase = before ? phase : "after";
    const frame = visiblePhase === "after" ? after : before;
    const entity = entities.get(selected);
    byId("shot-key").hidden = visiblePhase !== "after" || !after.shots?.length;
    for (const button of phaseButtons) {
      button.disabled = !before && button.dataset.phase !== "after";
      button.setAttribute("aria-pressed", String(button.dataset.phase === visiblePhase));
    }
    for (const [team, name] of teamClasses.entries()) {
      const counts = [];
      for (const [rows, perTeam, label] of [[frame.alive, dronesPerTeam, "drone"], [frame.hp, basesPerTeam, "base"]]) {
        const count = rows.slice(team * perTeam, (team + 1) * perTeam).filter(Boolean).length;
        counts.push(`${count}/${quantity(perTeam, label)}`);
      }
      byId(`${name}-assets`).textContent = counts.join(" · ");
    }
    if (size.width <= 66 || size.height <= 44) return;
    const projection = frameProjection();
    const { point } = projection;
    arena.setAttribute("viewBox", `0 0 ${size.width} ${size.height}`);
    for (const layer of ["grid", "cone", "shots", "move", "entities"]) byId(`arena-${layer}`).replaceChildren();
    document.querySelector("#arena-clip").replaceChildren();
    renderGrid(projection);
    if (entity.kind === "drone" && alive(entity, frame)) renderCone(byId("arena-cone"), entity, frame, projection);
    if (visiblePhase === "after") renderShots(byId("arena-shots"), frame, point);
    renderMovement(byId("arena-move"), entity, before, after, point, visiblePhase);
    renderEntities(byId("arena-entities"), frame, point);
  }

  function stateRows(entity, frame) {
    const active = alive(entity, frame);
    return entity.kind === "drone" ? [
      ["Status", active ? "Alive" : "Eliminated"],
      ["Position", coord(position(entity, frame))],
      ["Heading", headingLabel(frame.heading[entity.index])],
    ] : [
      ["Status", active ? "Standing" : "Destroyed"],
      ["Position", coord(position(entity, frame))],
      ["HP", `${frame.hp[entity.index]} / ${replay.baseMaxHp[entity.index]}`],
      ["Point value", number(replay.baseValue[entity.index])],
    ];
  }

  function renderInspector() {
    const entity = entities.get(selected);
    const { index, kind, label, key } = entity;
    const after = frames[turn];
    const before = frames[turn - 1];
    picker.value = key;
    byId("transition").textContent = before ? `Turn ${before.t} → ${after.t}` : "Initial state";
    byId("before-column").hidden = !before;
    byId("after-column").textContent = before ? "After" : "Initial";
    const beforeRows = before ? stateRows(entity, before) : [];
    const rows = stateRows(entity, after).map(([name, value], rowIndex) => {
      const row = document.createElement("tr");
      const header = document.createElement("th");
      header.scope = "row";
      header.textContent = name;
      row.append(header);
      if (before) row.insertCell().textContent = beforeRows[rowIndex][1];
      row.insertCell().textContent = value;
      return row;
    });
    byId("entity-state").replaceChildren(...rows);

    const move = movement(entity, before, after);
    const action = byId("policy-action");
    byId("actual-move").textContent = "";
    if (kind === "base") {
      action.textContent = "Bases do not take actions.";
    } else if (!before) {
      action.textContent = "The first action appears on turn 1.";
    } else if (!move) {
      action.textContent = "Already eliminated before this turn.";
    } else {
      if (move.action) {
        const [direction, turnAction] = move.action;
        const magnitude = length(move.requested);
        const command = direction ? `Move ${replay.moveNames[direction].toLowerCase()} ${quantity(magnitude, "unit")}` : "Hover";
        const rotation = ["clockwise", "hold heading", "counterclockwise"][turnAction];
        const turnLabel = turnAction === 1 ? rotation : `turn ${number(replay.turnDegrees)}° ${rotation}`;
        action.textContent = `${command} · ${turnLabel}`;
      } else {
        action.textContent = "Command unavailable in this recording.";
      }
      if (move.actual) {
        const distance = length(move.actual);
        byId("actual-move").textContent = `Actual displacement ${coord(move.actual)} · ${quantity(distance, "unit")}${move.clipped ? " · clipped by boundary" : ""}`;
      }
    }

    const descriptions = [];
    for (const [shooter, targetKind, targetIndex, probability, hit] of after.shots ?? []) {
      const from = entityAt("drone", shooter);
      const to = entityAt(targetKind, targetIndex);
      if (from.key === key || to.key === key) {
        descriptions.push(`${from.label} → ${to.label} · ${hit ? "hit" : "miss"} (${number(probability * 100)}% chance)`);
      }
    }
    if (kind === "drone" && after.dronesKilled?.[index]) descriptions.push(`${label} eliminated.`);
    if (kind === "base" && after.basesHit?.[index]) {
      descriptions.push(`${label} HP ${before.hp[index]} → ${after.hp[index]}${after.basesDestroyed[index] ? " · destroyed" : ""}.`);
    }
    if (!descriptions.length) descriptions.push(before ? "No automatic events for this entity." : "No turn has resolved.");
    byId("automatic-events").replaceChildren(...descriptions.map(description => {
      const item = document.createElement("li");
      item.textContent = description;
      return item;
    }));
  }

  function renderTimeline() {
    const frame = frames[turn];
    const eventLabels = Object.values(events[turn]).flat();
    slider.value = turn;
    byId("turn-label").textContent = `Turn ${frame.t} of ${frames[lastTurn].t}`;
    slider.setAttribute("aria-valuetext", byId("turn-label").textContent);
    byId("turn-reward").textContent = score(frame.reward);
    byId("running-return").textContent = score(frame.cumulative);
    byId("turn-events").textContent = eventLabels.length ? eventLabels.join(" · ") :
      turn === 0 ? "Initial state" : "No base damage or eliminations this turn.";
    byId("previous-turn").disabled = turn === 0;
    byId("next-turn").disabled = turn === lastTurn;
    byId("previous-event").disabled = !eventTurns.some(index => index < turn);
    byId("next-event").disabled = !eventTurns.some(index => index > turn);
    for (const cursor of eventCursors) cursor.style.setProperty("--position", eventPosition(turn));
    for (const marker of eventMarkers) {
      if (Number(marker.dataset.turn) === turn) marker.setAttribute("aria-current", "step");
      else marker.removeAttribute("aria-current");
    }
  }

  function render() {
    renderArena();
    renderInspector();
    renderTimeline();
  }

  function pause() {
    if (timer !== null) window.clearInterval(timer);
    timer = null;
    byId("play").textContent = "Play";
    byId("turn-events").setAttribute("aria-live", "polite");
  }

  function seek(index) {
    pause();
    turn = Math.min(lastTurn, Math.max(0, index));
    render();
  }

  function select(key) {
    selected = key;
    renderArena();
    renderInspector();
  }

  picker.addEventListener("change", () => select(picker.value));
  arena.addEventListener("click", event => {
    const entity = event.target.closest("[data-entity]");
    if (entity) select(entity.dataset.entity);
  });
  for (const button of phaseButtons) {
    button.addEventListener("click", () => {
      phase = button.dataset.phase;
      renderArena();
    });
  }
  slider.max = lastTurn;
  slider.disabled = lastTurn === 0;
  slider.addEventListener("input", () => seek(Number(slider.value)));
  byId("previous-turn").addEventListener("click", () => seek(turn - 1));
  byId("next-turn").addEventListener("click", () => seek(turn + 1));
  byId("previous-event").addEventListener("click", () => seek(eventTurns.filter(index => index < turn).at(-1)));
  byId("next-event").addEventListener("click", () => seek(eventTurns.find(index => index > turn)));
  byId("play").disabled = lastTurn === 0;
  byId("play").addEventListener("click", () => {
    if (timer !== null) {
      pause();
      return;
    }
    if (turn === lastTurn) {
      turn = 0;
      render();
    }
    byId("play").textContent = "Pause";
    byId("turn-events").setAttribute("aria-live", "off");
    timer = window.setInterval(() => {
      turn++;
      if (turn === lastTurn) pause();
      render();
    }, 220);
  });
  new ResizeObserver(([entry]) => {
    size.width = entry.contentRect.width;
    size.height = entry.contentRect.height;
    renderArena();
  }).observe(arena);
  render();
})();
