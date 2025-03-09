import time
import json
import logging
import asyncio
import os
from openai import AsyncAzureOpenAI
import httpx
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from backend.settings import app_settings, MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
from quart import jsonify, request
from backend.sjc.assistant_filter_bullhorn_candidates_with_functions.assistant import call_azure_logic_app, init_openai_client


DEBUG = os.environ.get("DEBUG", "false")


async def init_assistant_client():
    try:
        azure_openai_client = await init_openai_client()

        # Lesen des Prompts aus der Datei
        with open(os.path.join(os.path.dirname(__file__), 'system_message.txt'), 'r', encoding='utf-8') as file:
            system_message = file.read()

        # Datei mit Kandidaten-Metadaten laden
        file_path = "./back"
        with open(os.path.join(os.path.dirname(__file__), 'candidates-meta.json'), "r", encoding="utf-8") as file:
            candidates_meta = json.load(file)

        # System-Message mit den Kandidaten-Metadaten erweitern
        system_message += f"\n\n📋 **Kandidaten-Feld-Beschreibung:**\n{json.dumps(candidates_meta, indent=2, ensure_ascii=False)}"

        assistant = await azure_openai_client.beta.assistants.create(
            model="gpt-4o",
            instructions=system_message,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "bullhorn_search",
                        "description": "Durchsucht Bullhorn nach Kandidaten anhand einer Suchanfrage.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "searchQ": {"type": "string", "description": "Die Suchanfrage für Bullhorn"}
                            }
                        },
                        "strict": False,
                    }
                }
            ],
            tool_resources={},
            temperature=1,
            top_p=1,
        )

        logging.debug(f"Assistant created with ID: {assistant.id}")
        logging.debug(f"Registered tools: {assistant.tools}")

        return assistant
    except Exception as e:
        logging.exception(f"Exception in Assistant Initialization {e}")
        raise e


async def assistant_action_bullhorn(azure_openai_client, thread_id, tool_call, run):
    """
    Holt Kandidaten aus Bullhorn und speichert sie im Thread.
    """
    try:
        # 🔥 Bullhorn durchsuchen
        logging.info(
            f"Bullhorn-Suche wird durchgeführt: {tool_call.function.arguments}")
        arguments = json.loads(tool_call.function.arguments)
        tool_result = await call_azure_logic_app({"searchQ": arguments["searchQ"]})

        # 🔥 Check, ob die Kandidaten zurückgegeben wurden
        if not tool_result or "candidates" not in tool_result:
            logging.error(
                "Keine Kandidaten in der Antwort von Bullhorn gefunden.")
            raise ValueError("Bullhorn-Suche hat keine Kandidaten gefunden.")

        # **Hier wird die Antwort zusammengefasst**
        summary_message = f"Es wurden {tool_result['count']} von {tool_result['total']} Kandidaten gefunden."
        logging.info(f"Tool result summary: {summary_message}")
        candidates = tool_result["candidates"]
        full_json_str = json.dumps(tool_result, ensure_ascii=False)

        if DEBUG.lower() == "true":
            with open("./bullhorn_candidates.json", "w", encoding="utf-8") as f:
                f.write(full_json_str)

        # **Ergebnis speichern (reduzierte Antwort)**
        await azure_openai_client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run.id,
            tool_outputs=[
                {
                    "tool_call_id": tool_call.id,
                    "output": full_json_str
                }
            ]
        )
        logging.debug(
            f"✅ Tool-Outputs erfolgreich gespeichert: {json.dumps({'candidates': candidates})[:40]}")

        # **Antwort an den Assistenten zurückgeben**
        return {"type": "bullhorn_results", "content": full_json_str}
    except Exception as e:
        logging.exception(f"Fehler in assistant_action_bullhorn: {e}")
        raise


async def assistant_action_handler(azure_openai_client, thread_id, run):
    """
    Behandelt 'requires_action'-Events für den Assistant.
    """
    tool_calls = run.required_action.submit_tool_outputs.tool_calls
    tool_call = tool_calls[0]
    requested_tool = tool_call.function.name

    logging.info(f"Run required actions: {run.required_action}")

    if requested_tool == "bullhorn_search":
        return await assistant_action_bullhorn(azure_openai_client, thread_id, tool_call, run)
    else:
        raise NotImplementedError(
            f"Tool '{requested_tool}' ist nicht implementiert.")


async def wait_for_run_completion(azure_openai_client, thread_id, run_id, timeout=120):
    """
    Wartet auf die Fertigstellung des Assistant-Runs mit Timeout.
    Falls 'requires_action' auftritt, wird direkt assistant_action_handler() aufgerufen.
    """
    start_time = asyncio.get_event_loop().time()
    requires_action_handled = False  # Markiert, ob bereits eine Action gestartet wurde
    completed_actions = []  # ❗ Speichert alle nach dem Run benötigten Aktionen

    while True:
        if asyncio.get_event_loop().time() - start_time > timeout:
            raise TimeoutError(
                "Der Assistant-Run hat das Timeout überschritten.")

        run = await azure_openai_client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )

        if run.status in ['completed', 'failed']:
            if run.status == 'completed':
                return run
            return run  # Erfolg oder Fehler

        if run.status == 'requires_action':
            if requires_action_handled:
                raise RuntimeError(
                    "Mehr als eine 'requires_action' erkannt! Mögliche Endlosschleife.")

            action_result = await assistant_action_handler(azure_openai_client, thread_id, run)

            if action_result:
                # Speichert Rückgaben für später
                completed_actions.append(action_result)

            continue  # Zurück zur Schleife, um den neuen Status zu prüfen

        await asyncio.sleep(2)


async def filter_bullhorn_candidates_with_context():
    request_json = await request.get_json()
    req_user_input = request_json.get(
        "content", "Bitte durchsuche Bullhorn nach Managern aus der Solar-Branche.")
    req_thread_id = request_json.get("thread_id", None)
    req_assistant_id = request_json.get("assistant_id", None)
    logging.debug(
        f"filter_bullhorn_candidates_with_context User Input: {req_user_input} Thread ID: {req_thread_id} Assistant ID: {req_assistant_id} {request_json}")

    azure_openai_client = await init_openai_client()
    if not req_assistant_id:
        assistant = await init_assistant_client()
    else:  # Wiederverwenden des vorhandenen Assistenten
        assistant = await azure_openai_client.beta.assistants.retrieve(assistant_id=req_assistant_id)

    # Thread und Nachricht erstellen
    if not req_thread_id:
        thread = await azure_openai_client.beta.threads.create()
    else:
        thread = await azure_openai_client.beta.threads.retrieve(thread_id=req_thread_id)

    await azure_openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=req_user_input
    )

    # Thread ausführen
    run = await azure_openai_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # Warten, bis der Run abgeschlossen ist oder eine Aktion erforderlich ist
    run = await wait_for_run_completion(azure_openai_client, thread.id, run.id)

    logging.info(
        f"✅ Tool-Nachricht erfolgreich im Thread - req-thread-id:{req_thread_id}/thread-id:{thread.id} - gespeichert.")
    messages = await azure_openai_client.beta.threads.messages.list(thread_id=thread.id)
    for msg in messages.data:
        logging.info(f"Thread Nachricht - Role: {msg.role}, Inhalt: {msg.content[:200]}")  # Nur die ersten 200 Zeichen

    if run.status == 'completed':
        messages = await azure_openai_client.beta.threads.messages.list(thread_id=thread.id)
        # Die letzte Nachricht holen
        if messages.data:
            # OpenAI gibt die Nachrichten in absteigender Reihenfolge zurück (neueste zuerst)
            last_message = messages.data[0]
        else:
            last_message = None

        return jsonify({
            "id": thread.id,
            "assistant_id": assistant.id,
            "object": "chat.completion",
            "created": thread.created_at,
            "model": assistant.model,
            "choices": [
                {
                    "messages": [
                        {
                            "role": last_message.role,
                            "content": "".join([block.text.value for block in last_message.content]),
                            "id": last_message.id,
                            "created_at": last_message.created_at
                        }
                    ] if last_message else []
                }
            ]
        }), 200
    else:
        logging.error(
            f"Run failed: {run.status} {json.dumps(run.model_dump())}")
        return jsonify({
            "error": "Run failed",
            "status": run.status,
            "details": json.dumps(run.model_dump())
        }), 500
